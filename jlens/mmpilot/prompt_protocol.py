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

# MODALITIES and PROMPT_PROTOCOL_VERSION are re-exported, not restated. A
# modality this module accepted and the backend did not would be a prompt
# nothing can run, and the protocol string is in every completed fingerprint.
from jlens.mmpilot.backend import MODALITIES, text_hash
from jlens.mmpilot.capability import PROMPT_PROTOCOL_VERSION, build_question
from jlens.mmpilot.coordinate_swap import PROMPT_BOUNDARY_RULE
from jlens.mmpilot.store import payload_checksum

# ------------------------------------------------------------------ versions

#: The legacy, completed protocol. Its identifier is new; the *prompt* it names
#: is byte-for-byte the one :mod:`jlens.mmpilot.capability` has always built.
CANDIDATE_LISTED_IDENTIFICATION = "mmpilot.candidate_listed_identification.v1"

#: Open identification **restricted to animals**. Its question says "animal", so
#: it is only a valid ask when every identity in play is one.
OPEN_ANIMAL_IDENTIFICATION = "mmpilot.open_animal_identification.v1"

#: Open identification over **any** object category. Domain-neutral question,
#: mixed candidate sets allowed — and no property or multi-hop claim.
OPEN_ENTITY_IDENTIFICATION = "mmpilot.open_entity_identification.v1"

#: The downstream property, named for the property it actually asks about.
OPEN_ANIMAL_LEGS = "mmpilot.open_animal_legs.v1"

#: The same property with both entity labels hidden everywhere.
HIDDEN_ANIMAL_LEGS = "mmpilot.hidden_animal_legs.v1"

PROTOCOLS: tuple[str, ...] = (
    CANDIDATE_LISTED_IDENTIFICATION,
    OPEN_ANIMAL_IDENTIFICATION,
    OPEN_ENTITY_IDENTIFICATION,
    OPEN_ANIMAL_LEGS,
    HIDDEN_ANIMAL_LEGS,
)

#: Protocols whose instruction carries no candidate list.
OPEN_PROTOCOLS: tuple[str, ...] = (
    OPEN_ANIMAL_IDENTIFICATION,
    OPEN_ENTITY_IDENTIFICATION,
    OPEN_ANIMAL_LEGS,
    HIDDEN_ANIMAL_LEGS,
)

#: Identifiers this module used briefly and no longer accepts, with what to use
#: instead. They were domain-blind names on domain-specific questions — an
#: ``open_identification.v1`` that asked "what **animal** is present" cannot
#: screen ``toilet``, and a "how many legs" protocol called
#: ``open_downstream_property`` implies a generality it does not have. No real
#: run was ever recorded under any of them (they were MOCK-only), so they are
#: **renamed rather than deprecated**, and a caller that still names one gets a
#: refusal that says which protocol it meant.
RETIRED_PROTOCOLS: dict[str, str] = {
    "mmpilot.open_identification.v1": OPEN_ANIMAL_IDENTIFICATION,
    "mmpilot.open_downstream_property.v1": OPEN_ANIMAL_LEGS,
    "mmpilot.hidden_intermediate.v1": HIDDEN_ANIMAL_LEGS,
}

# ------------------------------------------------------------- task domains

#: COCO's own ``animal`` supercategory.
DOMAIN_ANIMAL = "animal"

#: The universal domain: any object category, mixed sets included. It is not a
#: supercategory — it is the statement that the question does not restrict one.
DOMAIN_ENTITY = "entity"

TASK_DOMAINS: tuple[str, ...] = (DOMAIN_ANIMAL, DOMAIN_ENTITY)

#: The property a protocol scores, or ``None`` for an identification protocol.
#: Versioned separately from the protocol because the answer *registry* can be
#: corrected without the question changing, and a run scored under a different
#: registry is a different measurement.
PROPERTY_LEG_COUNT = "animal_leg_count.v1"

PROPERTY_SCHEMAS: tuple[str, ...] = (PROPERTY_LEG_COUNT,)

#: How concepts are chosen for the first real animal-only study.
ANIMAL_CONCEPT_SELECTION_VERSION = "mmpilot.animal_concept_selection.v1"

#: The string completed runs recorded as ``prompt_protocol`` /
#: ``capability_protocol``. **Re-exported, never redefined** — it is imported
#: from the module that owns it, because changing it would change
#: :func:`jlens.mmpilot.pipeline.scientific_fingerprint` and refuse every
#: completed run's resume.
LEGACY_CAPABILITY_PROMPT_PROTOCOL = PROMPT_PROTOCOL_VERSION

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

class PromptProtocolError(ValueError):
    """An invalid protocol request, evidence object, or candidate set."""


class PromptLeakageError(PromptProtocolError):
    """The registered deterministic audit refused this prompt.

    Raised instead of downgrading to a weaker protocol. A ``hidden_animal_legs``
    prompt whose transcript names the source is not an
    ``open_animal_identification`` prompt — it is a prompt that failed, and the
    caller has to fix the evidence.
    """


class TaskDomainError(PromptProtocolError):
    """A concept does not carry the domain this protocol's question presumes.

    The failure this exists for: ``open_animal_identification`` asks *"What
    animal is present?"*, and the pilot's six-concept set contains ``toilet``
    and ``microwave``. Scoring those under that question does not measure open
    identification — it measures whether the model will name a non-animal when
    asked for an animal, which is a different experiment with a different
    interpretation. Refused rather than run.
    """


class PropertyAnswerError(PromptProtocolError):
    """A concept has no unique registered answer for this protocol's property.

    Unregistered and ambiguous are both refusals. A leg count guessed at
    scoring time would silently decide the experiment's ground truth.
    """


# ------------------------------------------------------------------ questions

#: Open identification, **animal domain**. It names no specific animal and lists
#: nothing, and it is byte-identical across the three evidence modalities — but
#: it does presume the answer is an animal, which is why
#: :data:`OPEN_ANIMAL_IDENTIFICATION` refuses a non-animal candidate.
OPEN_ANIMAL_IDENTIFICATION_QUESTION = (
    "What animal is present in the evidence? Answer with the animal name.\n"
    "Answer:"
)

#: Open identification, **domain-neutral**. Presumes nothing about the category,
#: so a mixed candidate set (``bird``, ``toilet``, ``microwave``) is legitimate.
OPEN_ENTITY_IDENTIFICATION_QUESTION = (
    "What is present in the evidence? Answer with its name.\nAnswer:"
)

#: The leg-count question. Animal-specific by construction: "the animal" is in
#: the sentence, and a leg count is not a property every object has.
ANIMAL_LEGS_QUESTION = (
    "How many legs does the animal typically have? Answer with a number.\n"
    "Answer:"
)

DEFAULT_QUESTIONS: dict[str, str] = {
    OPEN_ANIMAL_IDENTIFICATION: OPEN_ANIMAL_IDENTIFICATION_QUESTION,
    OPEN_ENTITY_IDENTIFICATION: OPEN_ENTITY_IDENTIFICATION_QUESTION,
    OPEN_ANIMAL_LEGS: ANIMAL_LEGS_QUESTION,
    HIDDEN_ANIMAL_LEGS: ANIMAL_LEGS_QUESTION,
}

#: The task domain each protocol's question presumes, and the property it
#: scores. Both are bound into the fingerprint: a run asked under a different
#: domain, or scored against a different property schema, is not the same
#: measurement. ``None`` for the legacy protocol, whose question is built from
#: whatever candidate list it is given and presumes nothing.
PROTOCOL_TASK_DOMAIN: dict[str, str | None] = {
    CANDIDATE_LISTED_IDENTIFICATION: None,
    OPEN_ANIMAL_IDENTIFICATION: DOMAIN_ANIMAL,
    OPEN_ENTITY_IDENTIFICATION: DOMAIN_ENTITY,
    OPEN_ANIMAL_LEGS: DOMAIN_ANIMAL,
    HIDDEN_ANIMAL_LEGS: DOMAIN_ANIMAL,
}

PROTOCOL_PROPERTY_SCHEMA: dict[str, str | None] = {
    CANDIDATE_LISTED_IDENTIFICATION: None,
    OPEN_ANIMAL_IDENTIFICATION: None,
    OPEN_ENTITY_IDENTIFICATION: None,
    OPEN_ANIMAL_LEGS: PROPERTY_LEG_COUNT,
    HIDDEN_ANIMAL_LEGS: PROPERTY_LEG_COUNT,
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
        domain: The category domain this concept belongs to, spelled as COCO's
            supercategory spells it (``"animal"``, ``"appliance"``,
            ``"furniture"``). ``None`` means **unspecified**, which a
            domain-restricted protocol refuses rather than assuming.
    """

    name: str
    aliases: tuple[str, ...] = ()
    domain: str | None = None

    @property
    def surface_forms(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for form in (self.name, *self.aliases):
            key = normalize(form)
            if key:
                seen.setdefault(key, None)
        return tuple(seen)

    @property
    def is_animal(self) -> bool:
        return self.domain == DOMAIN_ANIMAL

    def to_dict(self) -> dict:
        return {"name": self.name, "aliases": list(self.aliases), "domain": self.domain}


#: Registered aliases. Deliberately conservative: plurals only. A wrong alias
#: would refuse a clean prompt, and a missing one is a stated limit rather than
#: a silent pass.
DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {
    "bear": ("bears",),
    "bird": ("birds",),
    "cat": ("cats",),
    "cow": ("cows",),
    "dog": ("dogs",),
    "elephant": ("elephants",),
    "giraffe": ("giraffes",),
    "horse": ("horses",),
    "sheep": (),  # already plural; COCO spells it this way and so do we
    "zebra": ("zebras",),
    "microwave": ("microwaves",),
    "toilet": ("toilets",),
    # Not COCO categories. Registered because they are the paper's own
    # hidden-intermediate example ("the animal that spins webs"), and the
    # hidden protocol needs real domains and real leg counts to refuse against.
    "ant": ("ants",),
    "spider": ("spiders",),
}

#: COCO's ``animal`` supercategory, spelled as COCO spells it. This is a *read*
#: of the dataset's own ontology, not a judgement about what an animal is.
COCO_ANIMAL_CATEGORIES: tuple[str, ...] = (
    "bear",
    "bird",
    "cat",
    "cow",
    "dog",
    "elephant",
    "giraffe",
    "horse",
    "sheep",
    "zebra",
)

#: ``concept -> COCO supercategory``, for the categories this project has
#: actually used. It is deliberately **small and explicit** rather than a guess
#: for all eighty: an unregistered concept resolves to ``None``, and a
#: domain-restricted protocol refuses ``None`` instead of assuming.
#:
#: The live alternative is :func:`domain_registry_from_universe`, which reads
#: the supercategories out of the local COCO annotation files. Prefer it when a
#: :class:`~jlens.mmpilot.concepts.CategoryUniverse` is in hand.
CONCEPT_DOMAINS: dict[str, str] = {
    **{name: DOMAIN_ANIMAL for name in COCO_ANIMAL_CATEGORIES},
    # Not COCO categories, and so never selectable from SpokenCOCO coverage.
    # They exist here only as the paper's hidden-intermediate example.
    "ant": DOMAIN_ANIMAL,
    "spider": DOMAIN_ANIMAL,
    # The two that made this whole distinction necessary: both were in the
    # pilot's six-concept set, and neither is an animal.
    "microwave": "appliance",
    "toilet": "furniture",
    "bus": "vehicle",
    "train": "vehicle",
    "pizza": "food",
}


def domain_registry_from_universe(universe: Any) -> dict[str, str]:
    """``{category: supercategory}`` read from a discovered COCO universe.

    Args:
        universe: A :class:`jlens.mmpilot.concepts.CategoryUniverse`, or
            anything exposing ``supercategories``.

    A category the annotation files give no supercategory for is **omitted**,
    not defaulted — it then resolves to an unspecified domain, which a
    domain-restricted protocol refuses.
    """
    supercategories = dict(getattr(universe, "supercategories", None) or {})
    return {
        str(name): str(value).casefold()
        for name, value in sorted(supercategories.items())
        if str(value).strip()
    }


def concept_spec(
    name: str,
    aliases: Sequence[str] | None = None,
    *,
    domain: str | None = None,
    domain_registry: Mapping[str, str] | None = None,
) -> ConceptSpec:
    """A :class:`ConceptSpec`, with its domain resolved rather than assumed.

    Args:
        aliases: Overrides :data:`DEFAULT_ALIASES` for this concept.
        domain: States the domain explicitly. Wins over the registry.
        domain_registry: ``{concept: domain}``; defaults to
            :data:`CONCEPT_DOMAINS`. Pass
            :func:`domain_registry_from_universe`'s result to resolve against
            the local annotation files instead of this module's small table.

    An unregistered concept gets ``domain=None`` — *unspecified*, which is a
    refusal under a domain-restricted protocol rather than a pass.
    """
    registered = tuple(aliases) if aliases is not None else DEFAULT_ALIASES.get(name, ())
    registry = CONCEPT_DOMAINS if domain_registry is None else domain_registry
    resolved = domain if domain is not None else registry.get(str(name))
    return ConceptSpec(
        name=str(name),
        aliases=tuple(str(a) for a in registered),
        domain=None if resolved is None else str(resolved),
    )


def aliases_checksum(*concepts: ConceptSpec | None) -> str:
    """Checksum over every registered surface form, so a fingerprint binds them."""
    payload = {
        spec.name: sorted(spec.surface_forms) for spec in concepts if spec is not None
    }
    return payload_checksum(payload)


def domain_registry_checksum(registry: Mapping[str, str] | None = None) -> str:
    """Checksum of the ``concept -> domain`` table a run resolved against."""
    registry = CONCEPT_DOMAINS if registry is None else registry
    return payload_checksum(
        {"domains": {str(k): str(v) for k, v in sorted(registry.items())}}
    )


# --------------------------------------------------------- property answers

#: ``concept -> the leg counts that concept can have``. A **one-element** tuple
#: is a unique registered answer and the only thing
#: :data:`OPEN_ANIMAL_LEGS` will run on; more than one is ambiguous and is
#: refused; absent is unregistered and is refused.
#:
#: Every entry here is a COCO ``animal`` category with an uncontested count.
#: There is deliberately **no ambiguous entry shipped** — inventing a fake one
#: to demonstrate the refusal would put a wrong fact in a registry that decides
#: an experiment's ground truth. The ambiguous path is exercised by passing an
#: explicit registry (see the tests), which is also how a real contested concept
#: would be declared.
ANIMAL_LEG_COUNTS: dict[str, tuple[int, ...]] = {
    # The paper's hidden-intermediate pair. Not COCO categories.
    "ant": (6,),
    "spider": (8,),
    "bear": (4,),
    "bird": (2,),
    "cat": (4,),
    "cow": (4,),
    "dog": (4,),
    "elephant": (4,),
    "giraffe": (4,),
    "horse": (4,),
    "sheep": (4,),
    "zebra": (4,),
}

#: How an integer leg count is spelled for the external scorer. Both the word
#: and the digit, because the model may produce either and the audit must refuse
#: either appearing in a prompt.
LEG_COUNT_SURFACES: dict[int, tuple[str, ...]] = {
    0: ("zero", "0"),
    2: ("two", "2"),
    4: ("four", "4"),
    6: ("six", "6"),
    8: ("eight", "8"),
}


def property_registry_checksum(
    registry: Mapping[str, Sequence[int]] | None = None,
    *,
    schema: str = PROPERTY_LEG_COUNT,
) -> str:
    """Checksum of the property-answer registry a run scored against."""
    registry = ANIMAL_LEG_COUNTS if registry is None else registry
    return payload_checksum(
        {
            "schema": schema,
            "answers": {
                str(k): sorted(int(v) for v in values)
                for k, values in sorted(registry.items())
            },
            "surfaces": {
                str(k): list(v) for k, v in sorted(LEG_COUNT_SURFACES.items())
            },
        }
    )


def resolve_leg_count(
    concept: str, *, registry: Mapping[str, Sequence[int]] | None = None
) -> int:
    """The concept's single registered leg count, or refuse.

    Raises:
        PropertyAnswerError: If the concept is unregistered, or registered with
            more than one possible count. Both are refusals: a guessed count
            would decide the experiment's ground truth silently, and an
            ambiguous one has no ground truth to decide.
    """
    registry = ANIMAL_LEG_COUNTS if registry is None else registry
    counts = registry.get(str(concept))
    if counts is None:
        raise PropertyAnswerError(
            f"{concept!r} has no registered leg count in the "
            f"{PROPERTY_LEG_COUNT} registry, so this protocol has no ground "
            "truth for it. Register the count deliberately, or use a concept "
            "that has one — it is not inferred."
        )
    unique = sorted({int(value) for value in counts})
    if len(unique) != 1:
        raise PropertyAnswerError(
            f"{concept!r} is registered with leg counts {unique}, which is "
            "ambiguous. A protocol that scores one number cannot be run on a "
            "concept that has several; refusing rather than picking one."
        )
    return unique[0]


def leg_count_surfaces(count: int) -> tuple[str, ...]:
    """Every registered spelling of ``count``, word and digit."""
    surfaces = LEG_COUNT_SURFACES.get(int(count))
    if not surfaces:
        raise PropertyAnswerError(
            f"leg count {count} has no registered surface forms; the scorer "
            "would have nothing to score and the audit nothing to refuse"
        )
    return surfaces


# ----------------------------------------------------------- the domain gate


def assert_task_domain(
    protocol: str,
    *,
    concepts: Sequence[ConceptSpec | None],
    external_candidates: Sequence[str] = (),
    domain_registry: Mapping[str, str] | None = None,
) -> dict:
    """Hold every identity in play to the domain this protocol's question presumes.

    Args:
        protocol: One of :data:`PROTOCOLS`.
        concepts: The source and target specs (``None`` entries are skipped).
        external_candidates: The **identities** the scorer will score. Their
            domains are resolved through ``domain_registry`` — they arrive as
            bare strings, so this is the only place they can be checked. Pass
            nothing for a property protocol: its candidates are answers
            (``two``, ``four``), not identities, and belong to
            :func:`assert_property_candidates`.
        domain_registry: ``{concept: domain}``; defaults to
            :data:`CONCEPT_DOMAINS`.

    Returns:
        ``{"task_domain", "resolved", "checked", "registry_checksum"}``.

    Raises:
        TaskDomainError: If the protocol restricts a domain and any source,
            target, or externally scored identity is in a different one, or in
            an unspecified one. ``DOMAIN_ENTITY`` restricts nothing and records
            what it observed instead.
    """
    if protocol not in PROTOCOL_TASK_DOMAIN:
        raise PromptProtocolError(f"unknown prompt protocol {protocol!r}; known: {PROTOCOLS}")
    required = PROTOCOL_TASK_DOMAIN[protocol]
    registry = CONCEPT_DOMAINS if domain_registry is None else domain_registry

    resolved: dict[str, str | None] = {}
    for spec in concepts:
        if spec is not None:
            resolved[spec.name] = spec.domain
    for name in external_candidates:
        resolved.setdefault(str(name), registry.get(str(name)))

    record = {
        "task_domain": required,
        "domain_restricted": required not in (None, DOMAIN_ENTITY),
        "resolved": dict(sorted(resolved.items())),
        "checked": sorted(resolved),
        "registry_checksum": domain_registry_checksum(registry),
    }
    if required in (None, DOMAIN_ENTITY):
        # Nothing to refuse. What was observed is recorded, because a mixed set
        # under a domain-neutral question is a fact worth having in the artifact.
        record["observed_domains"] = sorted(
            {value for value in resolved.values() if value}
        )
        record["unspecified_domain_concepts"] = sorted(
            name for name, value in resolved.items() if not value
        )
        return record

    wrong = sorted(
        (name, value) for name, value in resolved.items() if value and value != required
    )
    unspecified = sorted(name for name, value in resolved.items() if not value)
    if wrong or unspecified:
        parts = [f"{name} is domain {value!r}" for name, value in wrong]
        parts += [f"{name} has no registered domain" for name in unspecified]
        raise TaskDomainError(
            f"{protocol} asks a question in the {required!r} domain, so every "
            f"source, target and externally scored identity must be in it. "
            + "; ".join(parts)
            + f".\n\nThe question is {DEFAULT_QUESTIONS[protocol].splitlines()[0]!r} "
            f"— scoring something outside the {required!r} domain against it "
            "does not measure open identification, it measures what the model "
            f"says when asked for a {required} that is not there. Use "
            f"{OPEN_ENTITY_IDENTIFICATION} for a mixed set, or register the "
            "concept's domain if it really is in this one."
        )
    return record


def assert_property_candidates(
    protocol: str,
    external_candidates: Sequence[str],
    *,
    surfaces: Mapping[int, Sequence[str]] | None = None,
) -> dict:
    """Hold a property protocol's scored answers to its registered answer space.

    A property protocol's external candidates are *answers* — ``two``, ``four``
    — not identities, so they are checked here rather than by
    :func:`assert_task_domain`. An answer outside the registry would be scored
    against nothing: the audit could not refuse it in a prompt, and no source or
    target could ever be its ground truth.

    Raises:
        PropertyAnswerError: If any candidate is not a registered surface form
            for some leg count.
    """
    table = LEG_COUNT_SURFACES if surfaces is None else surfaces
    by_surface = {
        normalize(form): int(count)
        for count, forms in table.items()
        for form in forms
    }
    unknown = sorted(
        name for name in external_candidates if normalize(name) not in by_surface
    )
    if unknown:
        raise PropertyAnswerError(
            f"{protocol} scores {PROTOCOL_PROPERTY_SCHEMA[protocol]}, but "
            f"{unknown} are not registered answers for it (registered: "
            f"{sorted(by_surface)}). An unregistered answer has no ground truth "
            "and cannot be refused in a prompt; register it or drop it."
        )
    return {
        "property_schema": PROTOCOL_PROPERTY_SCHEMA[protocol],
        "candidate_counts": {
            str(name): by_surface[normalize(name)] for name in external_candidates
        },
    }


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

#: Leakage policy by *family*, so two protocols that differ only in task domain
#: cannot drift apart in what they refuse. ``refuse`` fails the audit;
#: ``record`` reports the finding and passes. There is no third action that
#: quietly rewrites the protocol.
LEAKAGE_FAMILIES: dict[str, dict[str, str]] = {
    "candidate_listed": {
        "instruction_candidate_leakage": RECORD,
        "candidate_enumeration_detected": RECORD,
        "source_in_visible_evidence": RECORD,
        "target_in_visible_evidence": RECORD,
        "source_in_audio_transcript": RECORD,
        "target_in_audio_transcript": RECORD,
        "property_answer_in_prompt": RECORD,
        "semantic_filename_exposure": REFUSE,
    },
    "open_identification": {
        "instruction_candidate_leakage": REFUSE,
        "candidate_enumeration_detected": REFUSE,
        "source_in_visible_evidence": RECORD,
        "target_in_visible_evidence": REFUSE,
        "source_in_audio_transcript": RECORD,
        "target_in_audio_transcript": REFUSE,
        "property_answer_in_prompt": RECORD,
        "semantic_filename_exposure": REFUSE,
    },
    "open_property": {
        "instruction_candidate_leakage": REFUSE,
        "candidate_enumeration_detected": REFUSE,
        "source_in_visible_evidence": RECORD,
        "target_in_visible_evidence": REFUSE,
        "source_in_audio_transcript": RECORD,
        "target_in_audio_transcript": REFUSE,
        "property_answer_in_prompt": REFUSE,
        "semantic_filename_exposure": REFUSE,
    },
    "hidden_property": {
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

#: Which family each protocol belongs to. The task domain is orthogonal to the
#: leakage policy: ``open_animal_identification`` and
#: ``open_entity_identification`` refuse exactly the same things and differ only
#: in the domain their question presumes.
PROTOCOL_LEAKAGE_FAMILY: dict[str, str] = {
    CANDIDATE_LISTED_IDENTIFICATION: "candidate_listed",
    OPEN_ANIMAL_IDENTIFICATION: "open_identification",
    OPEN_ENTITY_IDENTIFICATION: "open_identification",
    OPEN_ANIMAL_LEGS: "open_property",
    HIDDEN_ANIMAL_LEGS: "hidden_property",
}

#: Per-protocol policy, derived from the families above.
LEAKAGE_POLICY: dict[str, dict[str, str]] = {
    protocol: dict(LEAKAGE_FAMILIES[family])
    for protocol, family in PROTOCOL_LEAKAGE_FAMILY.items()
}

#: Protocols that hide the intermediate entity entirely. Named once so a policy
#: lookup and a special case cannot disagree.
HIDDEN_PROTOCOLS: tuple[str, ...] = tuple(
    protocol
    for protocol, family in PROTOCOL_LEAKAGE_FAMILY.items()
    if family == "hidden_property"
)

#: Protocols that score a downstream property rather than an identity.
PROPERTY_PROTOCOLS: tuple[str, ...] = tuple(
    protocol for protocol, schema in PROTOCOL_PROPERTY_SCHEMA.items() if schema
)


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
    if policy["property_answer_in_prompt"] == REFUSE and protocol in HIDDEN_PROTOCOLS:
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
        basename = str(media_reference).replace("\\", "/").rsplit("/", 1)[-1]
        # Ordered dedup, not a set: this list decides the order of the recorded
        # matches, and a set's iteration order varies with PYTHONHASHSEED.
        reference_forms = list(
            dict.fromkeys(form for form in (basename, basename.rsplit(".", 1)[0]) if form)
        )
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
    task_domain: dict
    property_schema: str | None = None
    #: Checksum of the property-answer registry **actually used**, not of the
    #: module default — a run scored against a corrected registry is a different
    #: measurement even when the prompt is byte-identical.
    property_registry_checksum: str | None = None
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
            "task_domain": dict(self.task_domain),
            "property_schema": self.property_schema,
            "property_registry_checksum": self.property_registry_checksum,
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
    domain_registry: Mapping[str, str] | None = None,
    leg_counts: Mapping[str, Sequence[int]] | None = None,
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
        source / target: Concept specs, for auditing only. A domain-restricted
            protocol additionally holds them to its task domain.
        property_answers: ``{"source": ..., "target": ...}`` surface forms. For
            a property protocol this is **derived** from the registry when
            omitted, and refused rather than guessed when the registry has no
            unique answer.
        encode_candidate: ``str -> [token ids]``. Supplied, every candidate's
            **complete** token sequence is recorded here and handed to the
            scorer; the leading space is added exactly as
            :func:`jlens.mmpilot.capability.candidate_token_ids` adds it.
        encode_prompt: ``str -> [token ids]`` for the model-visible text.
        legacy_candidate_list: Candidate-listed protocol only. The concepts the
            legacy question enumerates.
        domain_registry: ``{concept: domain}`` for resolving the externally
            scored candidates' domains; defaults to :data:`CONCEPT_DOMAINS`.
        leg_counts: Overrides :data:`ANIMAL_LEG_COUNTS`.
        strict: Refuse a prompt the audit rejects. Turning it off returns the
            record instead of raising and is for *inspecting* a refusal, never
            for running one.

    Raises:
        PromptProtocolError: On an unknown or retired protocol, a candidate list
            supplied to an open protocol, or an open question that interpolates
            a candidate.
        TaskDomainError: When a domain-restricted protocol is handed a concept
            from another domain, or one whose domain is unspecified.
        PropertyAnswerError: When a property protocol's source or target has no
            unique registered answer.
        PromptLeakageError: When ``strict`` and the audit refuses.
    """
    if protocol in RETIRED_PROTOCOLS:
        raise PromptProtocolError(
            f"prompt protocol {protocol!r} was renamed to "
            f"{RETIRED_PROTOCOLS[protocol]!r}. The old name was domain-blind on a "
            "domain-specific question — an 'open identification' protocol that "
            "asks 'what ANIMAL is present' cannot screen 'toilet'. Use the new "
            f"name, or {OPEN_ENTITY_IDENTIFICATION!r} for a mixed category set."
        )
    if protocol not in PROTOCOLS:
        raise PromptProtocolError(f"unknown prompt protocol {protocol!r}; known: {PROTOCOLS}")
    names = _normalize_candidates(external_candidates)

    # The domain gate runs before anything is rendered: a prompt that should not
    # exist is never built, hashed, or handed to a scorer. A property protocol's
    # candidates are answers rather than identities, so only its source and
    # target are domain-checked; the answers go through the property registry.
    identity_candidates = (
        ()
        if protocol == CANDIDATE_LISTED_IDENTIFICATION
        or PROTOCOL_PROPERTY_SCHEMA[protocol] is not None
        else names
    )
    domain_record = assert_task_domain(
        protocol,
        concepts=(source, target),
        external_candidates=identity_candidates,
        domain_registry=domain_registry,
    )
    if PROTOCOL_PROPERTY_SCHEMA[protocol] is not None:
        domain_record["property_candidates"] = assert_property_candidates(
            protocol, names
        )

    if protocol == CANDIDATE_LISTED_IDENTIFICATION:
        if question is None:
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

    property_schema = PROTOCOL_PROPERTY_SCHEMA[protocol]
    if property_schema is not None:
        if source is None or target is None:
            raise PromptProtocolError(
                f"{protocol} scores a property of the source and of the target, "
                "so both concepts must be named. Without them there is no ground "
                "truth to score against and nothing for the audit to refuse."
            )
        if property_answers is None:
            # Derived, never guessed: resolve_leg_count refuses an unregistered
            # or ambiguous concept rather than inventing its ground truth.
            property_answers = {
                "source": leg_count_surfaces(
                    resolve_leg_count(source.name, registry=leg_counts)
                ),
                "target": leg_count_surfaces(
                    resolve_leg_count(target.name, registry=leg_counts)
                ),
            }
        elif not property_answers:
            raise PromptProtocolError(
                f"{protocol} was given an empty property_answers. Pass None to "
                f"derive them from the {property_schema} registry, or state them; "
                "an empty mapping would make the audit pass by omission."
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
        task_domain=domain_record,
        property_schema=property_schema,
        property_registry_checksum=(
            property_registry_checksum(leg_counts, schema=property_schema)
            if property_schema
            else None
        ),
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

    The task domain, the property schema, and the checksums of the registries
    they were resolved against are bound here too. Asking the same question of
    an animal-only set and of a mixed set is not the same experiment, and a
    corrected leg-count registry changes the ground truth a property run was
    scored against even when nothing about the prompt moves.
    """
    candidate_ids = built.external_candidate_token_ids or {}
    domain = dict(built.task_domain or {})
    payload = {
        "prompt_protocol_version": built.protocol_version,
        "task_domain": domain.get("task_domain"),
        "task_domain_restricted": bool(domain.get("domain_restricted")),
        "concept_domains": dict(domain.get("resolved") or {}),
        "domain_registry_checksum": domain.get("registry_checksum"),
        "property_schema": built.property_schema,
        "property_registry_checksum": built.property_registry_checksum,
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
    OPEN_ANIMAL_IDENTIFICATION: "open_cross_modal_animal_identification",
    OPEN_ENTITY_IDENTIFICATION: "open_cross_modal_entity_identification",
    OPEN_ANIMAL_LEGS: "animal_leg_count_recomputation",
    HIDDEN_ANIMAL_LEGS: "hidden_animal_multi_hop_reasoning",
}

#: What each protocol may never support, stated so a report cannot imply it.
#: Note what the two identification protocols exclude in *each other's*
#: direction: an animal-only result is not a general object-identification
#: result, and a mixed-set result is not evidence about animals specifically.
EXCLUDED_CLAIMS: dict[str, tuple[str, ...]] = {
    CANDIDATE_LISTED_IDENTIFICATION: (
        "spontaneous unprompted concept emergence",
        "open cross-modal identification",
        "downstream property recomputation",
        "multi-hop reasoning",
    ),
    OPEN_ANIMAL_IDENTIFICATION: (
        "general object identification outside the animal domain",
        "downstream property recomputation",
        "multi-hop reasoning",
    ),
    OPEN_ENTITY_IDENTIFICATION: (
        "animal-specific identification",
        "downstream property recomputation",
        "multi-hop reasoning",
    ),
    OPEN_ANIMAL_LEGS: (
        "a property claim outside the animal domain",
        "multi-hop reasoning",
    ),
    HIDDEN_ANIMAL_LEGS: ("a reasoning claim outside the animal domain",),
}


def protocol_claim_admissibility(
    *,
    protocol: str,
    leakage: Mapping | None,
    mode: str,
    task_domain: Mapping | None = None,
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
        task_domain: The record from :func:`assert_task_domain`, i.e.
            ``BuiltPrompt.task_domain``. Required for a domain-restricted
            protocol: a claim that says "animal" must be able to show that
            every identity in play was one.
        identity_replacement_passed: Whether the identity condition actually
            replaced the identity. Required by :data:`OPEN_ANIMAL_LEGS`.
        direct_answer_control_passed: Whether inserting the answer's own lens
            vector failed to reproduce the effect. Required by
            :data:`OPEN_ANIMAL_LEGS`.
        direct_answer_onset_control_passed: The onset version of the same
            control. Required by :data:`HIDDEN_ANIMAL_LEGS`.

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

    required_domain = PROTOCOL_TASK_DOMAIN[protocol]
    requirements["task_domain"] = required_domain
    if required_domain not in (None, DOMAIN_ENTITY):
        observed = dict(task_domain or {})
        requirements["task_domain_verified"] = observed.get("task_domain")
        if not observed:
            reasons.append(
                f"{protocol} restricts its identities to the {required_domain} "
                "domain, but no task-domain record was supplied, so that "
                "restriction cannot be shown to have held"
            )
        elif observed.get("task_domain") != required_domain:
            reasons.append(
                f"the task-domain record is for {observed.get('task_domain')!r}, "
                f"not {required_domain!r}"
            )
        else:
            offenders = sorted(
                name
                for name, value in (observed.get("resolved") or {}).items()
                if value != required_domain
            )
            if offenders:
                reasons.append(
                    f"{offenders} are not in the {required_domain} domain; a "
                    f"{required_domain} claim cannot rest on them"
                )

    if protocol == OPEN_ANIMAL_LEGS:
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
    if protocol == HIDDEN_ANIMAL_LEGS:
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
        "task_domains": dict(PROTOCOL_TASK_DOMAIN),
        "property_schemas": dict(PROTOCOL_PROPERTY_SCHEMA),
        "statements": [
            "A candidate_listed_identification result supports only a "
            "candidate-conditioned claim.",
            "An open_animal_identification result supports open cross-modal "
            "identification only when the target leakage checks pass and every "
            "source, target and externally scored identity is in the animal "
            "domain.",
            "An open_entity_identification result supports open cross-modal "
            "identification over whatever categories it scored, and never a "
            "property or multi-hop claim.",
            "An open_animal_legs result supports leg-count recomputation only "
            "when identity replacement also succeeds and the direct-answer "
            "controls pass.",
            "A hidden_animal_legs result supports multi-hop reasoning only when "
            "both entity names are absent under the registered deterministic "
            "audit and the direct-answer onset control passes.",
            "A domain-restricted protocol's claim is inadmissible without a "
            "task-domain record showing the restriction held.",
            "No MOCK result supports any scientific claim.",
        ],
    }
    return {**payload, "rule_checksum": payload_checksum(payload)}


# ------------------------------------------- predeclared animal concept set

#: Fields a ranking row must **not** carry. The animal concept set is chosen
#: before the model runs; a row carrying an accuracy or a prediction came from
#: after, and selecting on it would make the choice depend on the outcome.
POST_MODEL_ROW_FIELDS: tuple[str, ...] = (
    "accuracy",
    "n_correct",
    "candidate_scores",
    "capability",
    "correct",
    "prediction",
    "target_margin",
    "target_score",
)


def select_animal_concepts(
    ranked_rows: Sequence[Mapping],
    *,
    n_focal: int = 2,
    domain_registry: Mapping[str, str] | None = None,
    leg_counts: Mapping[str, Sequence[int]] | None = None,
    require_leg_counts: bool = True,
) -> dict:
    """The predeclared animal-only concept set for the first real swap study.

    Takes the rows :func:`jlens.mmpilot.expansion.rank_concepts` already
    produces — the existing deterministic ranking and evidence audit — and
    filters them. It **re-implements neither**: coverage is whatever the local
    SpokenCOCO/COCO annotation files actually support, and a concept that is not
    feasible there is dropped with the ranking's own ``unmet`` reasons attached.

    ``bird``, ``cat``, ``giraffe``, ``zebra``, ``sheep`` and ``cow`` are the
    *likely* survivors. They are not assumed: if the local data does not carry
    them, they do not appear in the result, and if fewer than ``n_focal + 1``
    animals survive, this refuses rather than filling the gap.

    Filters, applied in this order and all pre-model:

    1. domain — the concept resolves to :data:`DOMAIN_ANIMAL`;
    2. feasibility — the ranking row's own ``feasible`` flag;
    3. property — a unique registered leg count, when ``require_leg_counts``.

    Ranking order is preserved throughout and never re-sorted alphabetically,
    for the reason :func:`jlens.mmpilot.selection.select_focal_concepts` gives:
    the ranking *is* the deterministic pre-model statement of what the dataset
    supports best.

    Returns:
        ``{"selection_version", "animal_concepts", "focal", "non_focal",
        "excluded", "selection_checksum", ...}``. ``non_focal`` supplies the
        external unrelated control, which is why ``n_focal + 1`` are needed.

    Raises:
        PromptProtocolError: If a ranking row carries a post-model field, or if
            too few animal concepts survive.
    """
    rows = [dict(row) for row in ranked_rows]
    contaminated = sorted(
        {field for row in rows for field in POST_MODEL_ROW_FIELDS if field in row}
    )
    if contaminated:
        raise PromptProtocolError(
            f"the ranking rows carry post-model field(s) {contaminated}. The "
            "animal concept set is predeclared: choosing it from rows that "
            "already know how the model behaved would make the selection depend "
            "on the outcome, which is the failure this study is built to avoid."
        )

    registry = CONCEPT_DOMAINS if domain_registry is None else domain_registry
    kept: list[str] = []
    excluded: list[dict] = []
    for row in rows:
        name = str(row.get("concept", ""))
        if not name:
            raise PromptProtocolError(f"ranking row without a concept: {row}")
        domain = registry.get(name)
        if domain != DOMAIN_ANIMAL:
            excluded.append(
                {
                    "concept": name,
                    "stage": "domain",
                    "reason": (
                        f"domain {domain!r}, not {DOMAIN_ANIMAL!r}"
                        if domain
                        else "no registered domain"
                    ),
                }
            )
            continue
        if not row.get("feasible"):
            excluded.append(
                {
                    "concept": name,
                    "stage": "evidence_audit",
                    "reason": "; ".join(str(item) for item in row.get("unmet") or [])
                    or "not feasible in the local split",
                }
            )
            continue
        if require_leg_counts:
            try:
                resolve_leg_count(name, registry=leg_counts)
            except PropertyAnswerError as error:
                excluded.append(
                    {"concept": name, "stage": "property", "reason": str(error).splitlines()[0]}
                )
                continue
        kept.append(name)

    if len(kept) < n_focal + 1:
        raise PromptProtocolError(
            f"only {len(kept)} animal concept(s) survived the ranking and the "
            f"evidence audit ({kept}), which cannot supply {n_focal} focal "
            "concepts plus at least one external unrelated control. Coverage is "
            "a property of the local data, not something to relax: widen the "
            "manifest or lower n_focal deliberately.\n  excluded: "
            + "; ".join(f"{row['concept']} ({row['stage']})" for row in excluded)
        )

    payload = {
        "selection_version": ANIMAL_CONCEPT_SELECTION_VERSION,
        "task_domain": DOMAIN_ANIMAL,
        "property_schema": PROPERTY_LEG_COUNT if require_leg_counts else None,
        "require_leg_counts": bool(require_leg_counts),
        "n_focal": int(n_focal),
        "ranked_input": [str(row.get("concept")) for row in rows],
        "animal_concepts": list(kept),
        "focal": list(kept[:n_focal]),
        "non_focal": list(kept[n_focal:]),
        "leg_counts": {
            name: resolve_leg_count(name, registry=leg_counts) for name in kept
        }
        if require_leg_counts
        else {},
        "excluded": excluded,
        "domain_registry_checksum": domain_registry_checksum(registry),
        "property_registry_checksum": (
            property_registry_checksum(leg_counts) if require_leg_counts else None
        ),
        "order_rule": (
            "ranking order preserved; never re-sorted alphabetically, and never "
            "reordered by any model result"
        ),
    }
    return {**payload, "selection_checksum": payload_checksum(payload)}


__all__ = [
    "ANIMAL_CONCEPT_SELECTION_VERSION",
    "ANIMAL_LEG_COUNTS",
    "ANIMAL_LEGS_QUESTION",
    "AUDIT_LIMITS",
    "AUDIT_VERSION",
    "CANDIDATE_LISTED_IDENTIFICATION",
    "CANDIDATE_SCORING_VERSION",
    "CANDIDATE_VISIBILITY_RULE",
    "CLAIM_RULE_VERSION",
    "COCO_ANIMAL_CATEGORIES",
    "CONCEPT_DOMAINS",
    "DEFAULT_ALIASES",
    "DEFAULT_QUESTIONS",
    "DOMAIN_ANIMAL",
    "DOMAIN_ENTITY",
    "EXCLUDED_CLAIMS",
    "FORBIDDEN_BACKEND_KWARGS",
    "HIDDEN_ANIMAL_LEGS",
    "HIDDEN_PROTOCOLS",
    "LEAKAGE_CATEGORIES",
    "LEAKAGE_FAMILIES",
    "LEAKAGE_POLICY",
    "LEGACY_CAPABILITY_PROMPT_PROTOCOL",
    "LEG_COUNT_SURFACES",
    "MAXIMUM_CLAIM",
    "MODALITIES",
    "NORMALIZATION_RULE",
    "OPEN_ANIMAL_IDENTIFICATION",
    "OPEN_ANIMAL_IDENTIFICATION_QUESTION",
    "OPEN_ANIMAL_LEGS",
    "OPEN_ENTITY_IDENTIFICATION",
    "OPEN_ENTITY_IDENTIFICATION_QUESTION",
    "OPEN_PROTOCOLS",
    "OPEN_TEXT_EVIDENCE_TEMPLATE",
    "POST_MODEL_ROW_FIELDS",
    "PROPERTY_LEG_COUNT",
    "PROPERTY_PROTOCOLS",
    "PROPERTY_SCHEMAS",
    "PROTOCOLS",
    "PROTOCOL_LEAKAGE_FAMILY",
    "PROTOCOL_PROPERTY_SCHEMA",
    "PROTOCOL_TASK_DOMAIN",
    "RETIRED_PROTOCOLS",
    "TASK_DOMAINS",
    "BuiltPrompt",
    "ConceptSpec",
    "Evidence",
    "PromptLeakageError",
    "PromptProtocolError",
    "PropertyAnswerError",
    "TaskDomainError",
    "aliases_checksum",
    "assert_prompt_leakage_clean",
    "assert_task_domain",
    "audit_prompt_leakage",
    "backend_input_kwargs",
    "build_backend_inputs",
    "build_protocol_prompt",
    "claim_admissibility_rule_record",
    "concept_spec",
    "contains_surface",
    "domain_registry_checksum",
    "domain_registry_from_universe",
    "leg_count_surfaces",
    "looks_like_candidate_enumeration",
    "normalize",
    "prompt_protocol_fingerprint",
    "property_registry_checksum",
    "protocol_claim_admissibility",
    "resolve_leg_count",
    "select_animal_concepts",
]
