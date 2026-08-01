# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Where the pilot's candidate concepts come from, and what words express them.

The pilot used to screen six concepts — bus, cat, dog, horse, pizza, train —
chosen by hand before anyone looked at the data. That is an arbitrary starting
point: it is not derived from the dataset, it cannot be checked against the
dataset, and it silently decides the answer to the question the audit is
supposed to ask ("which concepts can this local copy actually support?").

SpokenCOCO has no concept ontology of its own. It inherits its visual semantics
from MS COCO, whose ``instances_*.json`` files define the object categories that
the images are actually annotated with. So the candidate universe is read from
those files, on this machine, with checksums recorded — and every category that
appears there is a candidate until the evidence says otherwise.

Two things live here:

:class:`CategoryUniverse`
    The categories discovered in the local ``instances_train*.json`` /
    ``instances_val*.json`` files, with their COCO ids, supercategories, the
    files they came from and those files' checksums.

:class:`LexicalSpec`
    For one category: the written forms that count as a mention of it, the
    forms that were *considered and rejected*, why in both cases, whether the
    category is lexically ambiguous, and which phrases void a match.

Matching stays what it was — normalized whole-word / whole-phrase regex against
an explicit list. No embeddings, no classifier, no language model, no fuzzy
similarity, no substring matching, no network call. What is new is that the
list is *derived* and *justified* rather than assumed.

The ambiguity policy, in one place
----------------------------------

Every category gets one of four statuses, and the ranking in
:mod:`jlens.mmpilot.expansion` reads it:

``clean``
    No non-object sense common in image captions. ``pizza``, ``giraffe``.

``resolved_by_exclusion``
    A real collision exists but a phrase exclusion removes it exactly.
    ``dog`` collides with ``hot dog``; excluding the phrase ``hot dog`` from
    ``dog``'s matches resolves it without weakening the term.

``alias_only``
    The bare category name is *not* usable — its dominant caption sense is not
    the object — but an unambiguous phrase is. COCO's ``remote`` means a remote
    control; the bare word is an adjective, so only ``remote control`` and its
    variants are accepted.

``ambiguous``
    A non-object sense is common enough to matter and cannot be excised by a
    phrase. ``train`` (the verb), ``bear`` (beyond ``teddy bear``). Accepted,
    flagged, and penalised in the ranking.

``excluded``
    No defensible lexical form, or the category is useless as a contrast.
    ``orange`` (the colour dominates); ``person`` (present in most of COCO, so
    it cannot discriminate and it poisons the matched-negative pool).

Nothing here is a guess about what a caption "probably means". Where a form was
rejected, the reason is recorded next to it and printed by the notebook.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from jlens.mmpilot.store import payload_checksum

#: Bumped when the discovery rule or the lexical table changes meaning. It is
#: part of every cache fingerprint, so old derived artifacts are refused rather
#: than silently reused under a new specification.
CONCEPT_UNIVERSE_VERSION = "jlens.mmpilot.concept_universe.v1"
LEXICAL_SPEC_VERSION = "jlens.mmpilot.lexical_spec.v1"

#: Filenames that hold COCO *object* annotations. ``captions_*.json`` is a
#: different file with no categories in it and is deliberately not matched.
INSTANCE_FILE_PREFIXES = ("instances_train", "instances_val")

AMBIGUITY_STATUSES = (
    "clean",
    "resolved_by_exclusion",
    "alias_only",
    "ambiguous",
    "excluded",
)

#: How much each status is trusted, for the ranking. Deterministic constants,
#: not tuned parameters.
AMBIGUITY_SCORE = {
    "clean": 1.0,
    "resolved_by_exclusion": 0.9,
    "alias_only": 0.75,
    "ambiguous": 0.4,
    "excluded": 0.0,
}


class CategoryDiscoveryError(RuntimeError):
    """No local COCO object-annotation file yielded a category universe."""


# ------------------------------------------------------------- morphology


#: Plurals English regular rules get wrong. Only entries a careful reader would
#: accept without argument; nothing here is a guess.
IRREGULAR_PLURALS: dict[str, tuple[str, ...]] = {
    "bus": ("buses", "busses"),
    "knife": ("knives",),
    "person": ("people", "persons"),
    "mouse": ("mice",),
    "sheep": ("sheep",),
}

#: Words that are already plural, so ``+s`` would be wrong.
ALREADY_PLURAL = frozenset({"skis", "scissors"})


def safe_plural(word: str) -> tuple[str, ...]:
    """Conservative English plurals for ``word``, possibly a multi-word phrase.

    Only the head (last) word is inflected: ``wine glass`` -> ``wine glasses``,
    never ``wines glass``. Returns an empty tuple when no rule applies safely
    rather than inventing one.
    """
    word = " ".join(str(word).lower().split())
    if not word:
        return ()
    prefix, head = word.rsplit(" ", 1) if " " in word else ("", word)
    if head in ALREADY_PLURAL:
        return ()
    irregular = IRREGULAR_PLURALS.get(head)
    if irregular:
        return tuple(f"{prefix} {form}".strip() for form in irregular if form != head)
    if head.endswith(("s", "x", "z", "ch", "sh")):
        plural = head + "es"
    # No ``-fe -> -ves`` rule: it is right for ``knife`` and wrong for
    # ``giraffe``, and there is no way to tell them apart by spelling. The one
    # COCO category it would help is listed in IRREGULAR_PLURALS instead.
    elif len(head) > 1 and head.endswith("y") and head[-2] not in "aeiou":
        plural = head[:-1] + "ies"
    else:
        plural = head + "s"
    return (f"{prefix} {plural}".strip(),)


# --------------------------------------------------------- the lexical table


@dataclass(frozen=True)
class LexicalSpec:
    """The written forms that count as a mention of one COCO category.

    Attributes:
        category: The COCO category name, verbatim as the annotation file
            spells it. This is also what :mod:`jlens.mmpilot.evidence` matches
            against for the *visual* half of the rule.
        terms: Accepted written forms. Multi-word entries are matched as whole
            phrases.
        exclusions: Phrases that void a match falling inside them. ``dog``
            excludes ``hot dog``, so "two hot dogs on a plate" is not a ``dog``
            mention.
        rejected: Forms that were considered and deliberately not accepted.
        rationale: ``form -> why``, for accepted and rejected forms alike.
        ambiguity: One of :data:`AMBIGUITY_STATUSES`.
        note: One sentence on the ambiguity, printed in the ranking table.
        eligible: False when the category may not be a pilot concept at all.
        derivation: ``"curated"`` or ``"default_morphology"`` — which path
            produced ``terms``, so an unreviewed category is visible as such.
    """

    category: str
    terms: tuple[str, ...]
    exclusions: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()
    rationale: Mapping[str, str] = field(default_factory=dict)
    ambiguity: str = "clean"
    note: str = ""
    eligible: bool = True
    derivation: str = "default_morphology"

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "terms": list(self.terms),
            "exclusions": list(self.exclusions),
            "rejected": list(self.rejected),
            "rationale": dict(sorted(self.rationale.items())),
            "ambiguity": self.ambiguity,
            "note": self.note,
            "eligible": self.eligible,
            "derivation": self.derivation,
        }

    @property
    def ambiguity_score(self) -> float:
        return AMBIGUITY_SCORE.get(self.ambiguity, 0.0)


#: Curated overrides, keyed by COCO category name. A category absent from this
#: table still becomes a candidate — it just gets its name plus a safe plural
#: and ``derivation="default_morphology"``. The table is *not* the universe.
#:
#: Each entry is ``(terms, exclusions, rejected, rationale, ambiguity, note)``.
#: Terms are additions to the category name itself, which is always accepted
#: unless the category is ``alias_only`` or ``excluded``.
_CURATED: dict[str, dict] = {
    # ---------------------------------------------------------- excluded
    "person": {
        "ambiguity": "excluded",
        "rejected": ("person", "people", "man", "woman"),
        "rationale": {
            "person": "annotated on the majority of COCO images, so it cannot "
            "discriminate between positives and matched negatives, and it "
            "would empty the negative pool",
        },
        "note": "ubiquitous co-occurring category; not a usable contrast",
    },
    "orange": {
        "ambiguity": "excluded",
        "rejected": ("orange", "oranges"),
        "rationale": {
            "orange": "in image captions the colour sense dominates ('an "
            "orange cat'), and nothing whole-word can separate it from the "
            "fruit",
            "oranges": "safe on its own, but accepting only the plural would "
            "make coverage depend on an accident of number",
        },
        "note": "colour sense dominates the fruit sense in captions",
    },
    # -------------------------------------------------------- alias only
    "remote": {
        "terms": ("remote control", "remote controls", "tv remote", "tv remotes"),
        "rejected": ("remote", "remotes"),
        "rationale": {
            "remote": "adjective in ordinary English ('a remote beach'); "
            "COCO's category is the handheld control",
            "remote control": "unambiguous phrase for exactly this object",
        },
        "ambiguity": "alias_only",
        "note": "bare 'remote' is an adjective; only the phrase is accepted",
    },
    "tie": {
        "terms": ("necktie", "neckties", "bow tie", "bow ties", "neck tie"),
        "rejected": ("tie", "ties", "tied"),
        "rationale": {
            "tie": "verb and sports sense ('the game ended in a tie') are at "
            "least as common as the garment",
            "necktie": "names the garment and nothing else",
        },
        "ambiguity": "alias_only",
        "note": "bare 'tie' is a verb / a drawn game; only the garment phrases count",
    },
    "mouse": {
        "terms": ("computer mouse", "computer mice", "wireless mouse"),
        "rejected": ("mouse", "mice"),
        "rationale": {
            "mouse": "COCO's category is the computer peripheral, but the "
            "animal sense dominates unqualified caption use",
            "computer mouse": "names the peripheral unambiguously",
        },
        "ambiguity": "alias_only",
        "note": "COCO's 'mouse' is the peripheral; the animal sense is excluded",
    },
    # ---------------------------------------------- resolved by exclusion
    "dog": {
        "terms": ("dogs", "puppy", "puppies"),
        "exclusions": ("hot dog", "hot dogs"),
        "rationale": {
            "puppy": "a puppy is a dog; no other sense in captions",
            "hot dog": "excluded phrase — 'two hot dogs' must not count as a "
            "dog mention, and 'hot dog' is its own COCO category",
        },
        "ambiguity": "resolved_by_exclusion",
        "note": "collides with the 'hot dog' category; the phrase is excluded",
    },
    "hot dog": {
        "terms": ("hot dogs", "hotdog", "hotdogs"),
        "rationale": {
            "hot dog": "phrase-matched, so it never fires on 'dog' alone",
            "hotdog": "the closed spelling of the same food",
        },
        "ambiguity": "resolved_by_exclusion",
        "note": "shares a token with 'dog'; both sides use phrase matching",
    },
    "bear": {
        "terms": ("bears",),
        "exclusions": ("teddy bear", "teddy bears"),
        "rejected": ("teddy bear",),
        "rationale": {
            "bear": "the animal; the verb sense ('bear left') is rare in "
            "image captions but is why this stays flagged",
            "teddy bear": "excluded phrase — a stuffed toy is a different "
            "COCO category and is not a bear",
        },
        "ambiguity": "ambiguous",
        "note": "'teddy bear' excluded; the verb sense remains a residual risk",
    },
    "car": {
        "terms": ("cars", "automobile", "automobiles"),
        "exclusions": ("train car", "train cars", "cable car", "cable cars"),
        "rejected": ("vehicle",),
        "rationale": {
            "automobile": "an exact synonym",
            "vehicle": "covers trucks, buses and motorcycles too, so it would "
            "produce caption evidence the annotation does not back",
            "train car": "excluded phrase — a carriage is part of a train",
        },
        "ambiguity": "resolved_by_exclusion",
        "note": "'train car' and 'cable car' excluded",
    },
    # ---------------------------------------------------------- ambiguous
    "train": {
        "terms": ("trains",),
        "rejected": ("training", "trained"),
        "rationale": {
            "train": "the vehicle; whole-word matching already rules out "
            "'training', but the bare verb ('they train daily') remains",
            "training": "verbal noun, never the vehicle",
        },
        "ambiguity": "ambiguous",
        "note": "the verb 'to train' shares the bare form",
    },
    "keyboard": {
        "terms": ("keyboards", "computer keyboard"),
        "rejected": ("piano",),
        "rationale": {
            "keyboard": "COCO means the computer keyboard; a musical keyboard "
            "is the same word",
            "piano": "a different instrument, not this category",
        },
        "ambiguity": "ambiguous",
        "note": "musical keyboards share the word",
    },
    "bicycle": {
        "terms": ("bicycles", "bike", "bikes"),
        "rationale": {
            "bike": "usually a bicycle in captions, but motorcycles are also "
            "called bikes — flagged rather than dropped, since dropping it "
            "would lose most real mentions",
        },
        "ambiguity": "ambiguous",
        "note": "'bike' is also used for motorcycles",
    },
    "sink": {
        "terms": ("sinks",),
        "rationale": {"sink": "the basin; the verb sense is rare in captions"},
        "ambiguity": "ambiguous",
        "note": "the verb 'to sink' shares the bare form",
    },
    # ------------------------------------------------- clean, with aliases
    "airplane": {
        "terms": ("airplanes", "plane", "planes", "aeroplane", "aeroplanes"),
        "rejected": ("jet", "jets", "aircraft"),
        "rationale": {
            "plane": "the ordinary caption word for this object",
            "jet": "also a colour and a verb; the gain is not worth it",
            "aircraft": "covers helicopters, which are not this category",
        },
    },
    "motorcycle": {
        "terms": ("motorcycles", "motorbike", "motorbikes"),
        "rejected": ("bike",),
        "rationale": {
            "bike": "claimed by 'bicycle', where it is the commoner reading",
        },
    },
    "couch": {
        "terms": ("couches", "sofa", "sofas"),
        "rationale": {"sofa": "the same object under its other common name"},
    },
    "tv": {
        "terms": ("tvs", "television", "televisions", "t v"),
        "rationale": {
            "t v": "what 'T.V.' normalizes to once punctuation is collapsed",
        },
    },
    "cell phone": {
        "terms": (
            "cell phones",
            "cellphone",
            "cellphones",
            "mobile phone",
            "mobile phones",
            "smartphone",
            "smartphones",
        ),
        "rejected": ("phone",),
        "rationale": {
            "phone": "also a landline, which is not this category",
        },
    },
    "dining table": {
        "terms": ("dining tables",),
        "rejected": ("table",),
        "rationale": {
            "table": "any surface; side tables and desks are not annotated as "
            "this category",
        },
    },
    "refrigerator": {
        "terms": ("refrigerators", "fridge", "fridges"),
        "rationale": {"fridge": "the ordinary short form"},
    },
    "donut": {
        "terms": ("donuts", "doughnut", "doughnuts"),
        "rationale": {"doughnut": "the other standard spelling"},
    },
    "hair drier": {
        "terms": ("hair driers", "hair dryer", "hair dryers", "hairdryer", "blow dryer"),
        "rationale": {"hair dryer": "COCO spells it 'drier'; captions do not"},
    },
    "tennis racket": {
        "terms": ("tennis rackets", "tennis racquet", "tennis racquets"),
        "rationale": {"tennis racquet": "the other standard spelling"},
    },
    "traffic light": {
        "terms": ("traffic lights", "stoplight", "stoplights", "traffic signal"),
        "rationale": {"stoplight": "the same fixture"},
    },
    "fire hydrant": {
        "terms": ("fire hydrants", "hydrant", "hydrants"),
        "rationale": {"hydrant": "no other sense in street photography"},
    },
    "wine glass": {
        "terms": ("wine glasses", "wineglass", "wineglasses"),
        "rejected": ("glass",),
        "rationale": {"glass": "the material, and any tumbler"},
    },
    "handbag": {
        "terms": ("handbags", "purse", "purses", "hand bag"),
        "rationale": {"purse": "the same object in American usage"},
    },
    "suitcase": {
        "terms": ("suitcases", "suit case"),
        "rejected": ("luggage", "bag"),
        "rationale": {
            "luggage": "a mass noun covering backpacks and handbags too",
        },
    },
    "surfboard": {
        "terms": ("surfboards", "surf board", "surf boards"),
        "rationale": {"surf board": "the open spelling"},
    },
    "backpack": {
        "terms": ("backpacks", "back pack", "rucksack", "rucksacks"),
        "rationale": {"rucksack": "the same object"},
    },
    "cow": {
        "terms": ("cows", "cattle"),
        "rationale": {"cattle": "the mass noun for the same animal"},
    },
    "sheep": {
        "terms": ("lamb", "lambs"),
        "rationale": {
            "sheep": "unchanged in the plural",
            "lamb": "a young sheep; the meat sense is not an image subject",
        },
    },
    "horse": {
        "terms": ("horses", "pony", "ponies"),
        "rationale": {"pony": "a horse"},
    },
    "cat": {
        "terms": ("cats", "kitten", "kittens", "kitty", "kitties"),
        "rejected": ("cattle",),
        "rationale": {
            "kitten": "a young cat",
            "cattle": "a different animal that merely shares three letters — "
            "whole-word matching already prevents it, and it is listed so the "
            "prevention is visible",
        },
    },
    "bus": {
        "terms": ("buses", "busses"),
        "rationale": {"busses": "the less common but standard plural"},
    },
    "skis": {
        "terms": ("ski",),
        "rejected": ("skiing",),
        "rationale": {
            "skis": "already plural",
            "ski": "the singular noun; whole-word matching keeps 'skiing' out",
            "skiing": "the activity, not the object",
        },
        "ambiguity": "ambiguous",
        "note": "'ski' is also a verb",
    },
    "potted plant": {
        "terms": ("potted plants", "houseplant", "houseplants"),
        "rejected": ("plant",),
        "rationale": {"plant": "any vegetation, and a factory"},
    },
    "sports ball": {
        "terms": ("sports balls",),
        "rejected": ("ball",),
        "rationale": {
            "ball": "also a dance, and COCO annotates only sports balls; "
            "captions almost never use the literal category phrase, so this "
            "category is expected to be infeasible on coverage rather than "
            "rescued by a loose term",
        },
    },
}


def lexical_spec(category: str) -> LexicalSpec:
    """The lexical specification for one discovered COCO category.

    Curated entries win; anything else gets the category name plus a safe
    plural and is marked ``derivation="default_morphology"`` so an unreviewed
    category is never mistaken for a reviewed one.
    """
    name = " ".join(str(category).lower().split())
    curated = _CURATED.get(name)
    if curated is None:
        terms = (name, *safe_plural(name))
        return LexicalSpec(
            category=name,
            terms=tuple(dict.fromkeys(terms)),
            rationale={
                name: "the COCO category name itself",
                **{p: "regular English plural of the category name" for p in safe_plural(name)},
            },
            derivation="default_morphology",
        )
    ambiguity = curated.get("ambiguity", "clean")
    extra = tuple(curated.get("terms", ()))
    if ambiguity in ("excluded", "alias_only"):
        terms = extra
    else:
        terms = (name, *extra)
    rationale = {
        name: f"the COCO category name itself ({ambiguity})",
        **dict(curated.get("rationale", {})),
    }
    for term in terms:
        rationale.setdefault(term, "accepted written form of this category")
    return LexicalSpec(
        category=name,
        terms=tuple(dict.fromkeys(terms)),
        exclusions=tuple(curated.get("exclusions", ())),
        rejected=tuple(curated.get("rejected", ())),
        rationale=rationale,
        ambiguity=ambiguity,
        note=curated.get("note", ""),
        eligible=ambiguity != "excluded",
        derivation="curated",
    )


# ----------------------------------------------------------- the universe


@dataclass(frozen=True)
class CategoryUniverse:
    """Every COCO object category found in the local annotation files.

    Attributes:
        categories: Category names, sorted, exactly as the files spell them.
        category_ids: ``name -> sorted COCO ids`` (ids agree across files, but
            they are recorded rather than assumed to).
        supercategories: ``name -> supercategory`` where the file provides one.
        sources: One entry per annotation file read: path, checksum, counts.
        specs: ``name -> LexicalSpec``.
    """

    categories: tuple[str, ...]
    category_ids: Mapping[str, tuple[int, ...]]
    supercategories: Mapping[str, str]
    sources: tuple[Mapping, ...]
    specs: Mapping[str, LexicalSpec]

    @property
    def eligible(self) -> tuple[str, ...]:
        """Categories that may be pilot concepts (``excluded`` ones dropped)."""
        return tuple(name for name in self.categories if self.specs[name].eligible)

    @property
    def excluded(self) -> tuple[str, ...]:
        return tuple(name for name in self.categories if not self.specs[name].eligible)

    def lexicon(self) -> dict[str, tuple[str, ...]]:
        """``{category: terms}`` for the eligible categories."""
        return {name: self.specs[name].terms for name in self.eligible}

    def coco_categories(self) -> dict[str, tuple[str, ...]]:
        """``{category: (category,)}`` — the visual half is the name itself."""
        return {name: (name,) for name in self.eligible}

    def exclusions(self) -> dict[str, tuple[str, ...]]:
        return {
            name: self.specs[name].exclusions
            for name in self.eligible
            if self.specs[name].exclusions
        }

    @property
    def universe_hash(self) -> str:
        """Hash of the discovered categories and their ids — no file paths."""
        return payload_checksum(
            {
                "version": CONCEPT_UNIVERSE_VERSION,
                "categories": list(self.categories),
                "category_ids": {k: list(v) for k, v in sorted(self.category_ids.items())},
            }
        )

    @property
    def lexical_hash(self) -> str:
        """Hash of every eligible category's full lexical specification."""
        return payload_checksum(
            {
                "version": LEXICAL_SPEC_VERSION,
                "specs": [self.specs[name].to_dict() for name in self.categories],
            }
        )

    def to_dict(self) -> dict:
        return {
            "universe_version": CONCEPT_UNIVERSE_VERSION,
            "lexical_spec_version": LEXICAL_SPEC_VERSION,
            "universe_hash": self.universe_hash,
            "lexical_hash": self.lexical_hash,
            "n_categories": len(self.categories),
            "categories": list(self.categories),
            "category_ids": {k: list(v) for k, v in sorted(self.category_ids.items())},
            "supercategories": dict(sorted(self.supercategories.items())),
            "eligible": list(self.eligible),
            "excluded": [
                {
                    "category": name,
                    "reason": self.specs[name].note,
                    "rationale": dict(sorted(self.specs[name].rationale.items())),
                }
                for name in self.excluded
            ],
            "sources": [dict(source) for source in self.sources],
            "specs": [self.specs[name].to_dict() for name in self.categories],
        }


def is_instance_annotation_path(path: str | os.PathLike[str]) -> bool:
    """Whether ``path`` looks like a COCO *object* annotation file.

    ``captions_train2014.json`` is not one: it has no ``categories``, and
    matching it here would silently substitute caption text for the visual
    half of the evidence rule.
    """
    name = Path(path).name.lower()
    return name.endswith(".json") and name.startswith(INSTANCE_FILE_PREFIXES)


def _payload_of(source) -> object:
    payload = getattr(source, "payload", None)
    if payload is not None:
        return payload
    return json.loads(Path(getattr(source, "path", source)).read_text(encoding="utf-8"))


def discover_category_universe(
    annotation_sources: Sequence,
    *,
    require_instance_files: bool = True,
) -> CategoryUniverse:
    """Read the candidate concept universe out of local COCO annotations.

    Args:
        annotation_sources: :class:`~jlens.mmpilot.expansion.MetadataSource`
            objects already classified as ``coco_object_annotation`` (or any
            object with ``.path`` and optionally ``.payload`` / ``.checksum``).
        require_instance_files: Prefer — and by default require — files named
            ``instances_train*.json`` / ``instances_val*.json``. When no such
            file is present, any source carrying a ``categories`` list is used
            instead and the fallback is recorded on the source entry.

    Raises:
        CategoryDiscoveryError: If no source yields a single category. The
            pilot cannot proceed without the visual half of the evidence rule,
            and guessing a universe would defeat the point of discovering one.
    """
    preferred = [s for s in annotation_sources if is_instance_annotation_path(getattr(s, "path", s))]
    used = preferred if (preferred or require_instance_files) else list(annotation_sources)
    if not used:
        used = list(annotation_sources)

    names: dict[str, set[int]] = {}
    supercategories: dict[str, str] = {}
    entries: list[dict] = []
    for source in used:
        path = str(getattr(source, "path", source))
        try:
            payload = _payload_of(source)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            entries.append({"path": path, "usable": False, "reason": f"unreadable: {exc}"})
            continue
        categories = payload.get("categories") if isinstance(payload, Mapping) else None
        if not isinstance(categories, list) or not categories:
            entries.append(
                {"path": path, "usable": False, "reason": "no 'categories' list"}
            )
            continue
        found = 0
        for item in categories:
            if not isinstance(item, Mapping) or "name" not in item:
                continue
            name = " ".join(str(item["name"]).lower().split())
            if not name:
                continue
            names.setdefault(name, set())
            try:
                names[name].add(int(item.get("id")))
            except (TypeError, ValueError):
                pass
            if item.get("supercategory"):
                supercategories.setdefault(name, str(item["supercategory"]).lower())
            found += 1
        annotations = payload.get("annotations")
        entries.append(
            {
                "path": path,
                "filename": Path(path).name,
                "checksum": getattr(source, "checksum", None),
                "usable": True,
                "is_instance_file": is_instance_annotation_path(path),
                "n_categories": found,
                "n_annotations": len(annotations) if isinstance(annotations, list) else 0,
                "selection": "instances_*.json" if preferred else "fallback: any file with categories",
            }
        )

    if not names:
        raise CategoryDiscoveryError(
            "no COCO object category could be read from any local annotation "
            "file, so the candidate concept universe is empty and the visual "
            "half of the evidence rule cannot be evaluated.\n"
            "Put COCO instances_train*.json / instances_val*.json (the object "
            "annotations, not captions_*.json) under the image root's "
            "annotations/ directory and re-run. Sources examined:\n"
            + json.dumps(entries, indent=2, default=str)
        )

    ordered = tuple(sorted(names))
    return CategoryUniverse(
        categories=ordered,
        category_ids={name: tuple(sorted(names[name])) for name in ordered},
        supercategories={k: v for k, v in sorted(supercategories.items())},
        sources=tuple(entries),
        specs={name: lexical_spec(name) for name in ordered},
    )


def format_lexical_table(universe: CategoryUniverse, *, limit: int | None = None) -> str:
    """The discovered universe with each category's status and terms."""
    header = f"{'category':16s} {'status':22s} {'terms':6s} exclusions / note"
    lines = [header, "-" * max(len(header), 72)]
    for name in universe.categories[:limit]:
        spec = universe.specs[name]
        detail = spec.note or ""
        if spec.exclusions:
            detail = f"excludes {list(spec.exclusions)}" + (f"; {detail}" if detail else "")
        lines.append(
            f"{name:16s} {spec.ambiguity:22s} {len(spec.terms):6d} {detail}"
        )
    if limit is not None and len(universe.categories) > limit:
        lines.append(f"... and {len(universe.categories) - limit} more")
    return "\n".join(lines)


__all__ = [
    "AMBIGUITY_SCORE",
    "AMBIGUITY_STATUSES",
    "CONCEPT_UNIVERSE_VERSION",
    "INSTANCE_FILE_PREFIXES",
    "LEXICAL_SPEC_VERSION",
    "CategoryDiscoveryError",
    "CategoryUniverse",
    "LexicalSpec",
    "discover_category_universe",
    "format_lexical_table",
    "is_instance_annotation_path",
    "lexical_spec",
    "safe_plural",
]
