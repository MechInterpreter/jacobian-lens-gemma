# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The final bounded rerun: the same populations, the unrestricted endpoint.

Two frozen study families inside one fingerprinted protocol.

**A — L33-L40 paper-style reasoning validation.** Exactly the completed
follow-up's design: the validated L33-L40 lens artifacts, the bands
``[33..40] [35..40] [38..40] [40]``, bird→cat and cat→bird, the intermediate
arm, the direct-answer arm as a positive control, text/image/spoken audio, α=1
primary with α=2 as a prespecified sensitivity, and the zero, alpha-matched
random and alpha-matched unrelated controls. The population is the one
``band3340_real_2a72bda9b4ba`` already selected, read out of its own immutable
report by group id. **This is a measurement-correction rerun on the same
population, not an independent replication**, and every artifact says so.

**B — canonical three-modality causal endpoint validation.** The completed
native-audio transfer run's own concepts, media, layers, directions, α values,
controls, prompts and lens artifacts, re-scored under the unrestricted endpoint.
Nothing is reselected from the corpus and no concept is substituted; if the exact
historical inputs cannot be reconstructed from immutable artifacts, the family is
refused rather than approximated.

What changes, and only this
===========================

Where the completed studies called
:func:`jlens.mmpilot.capability.score_candidate_sequences` and took an argmax
over the supplied candidates, this run calls
:func:`jlens.mmpilot.full_vocabulary.score_unrestricted_next_token` on the
untouched prompt and asks whether the target-appropriate token is the **global**
argmax over the whole vocabulary. The intervention, the hooks, the controls, the
population and the thresholds are the completed studies'. One forward pass per
trial instead of one per candidate — which is why a faithful rerun of a 10,752
candidate-pass design fits inside a 5,000 forward-pass cap.

Three verdict families that can never overwrite each other
==========================================================

The unrestricted-output verdict, the conditional-log-probability replication and
their cross-modal conjunction are computed separately, stored separately and
printed separately. A restricted-candidate preference, a positive margin without
global rank 1, a greedy completion, a direct-answer-arm success, one modality, or
a pooled direction can none of them produce a full-vocabulary GO.

Provenance is configured, never discovered
==========================================

Every completed artifact this study reads is named by an explicit path and
verified against a checksum **pinned in this module**. There is no globbing, no
"latest run", and no checksum accepted from the run being verified. A pin that is
empty is a refusal with a message naming the pin, not a default.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from jlens.mmpilot.band_swap import (
    BAND_ARMS,
    BAND_CONDITIONS,
    CONDITION_ALPHA,
    PRIMARY_ALPHA,
    SECONDARY_ALPHA,
    band_key,
)
from jlens.mmpilot.full_vocabulary import (
    ENDPOINT_CONDITIONAL_LOGPROB,
    ENDPOINT_GENERATION,
    ENDPOINT_RESTRICTED_CANDIDATE,
    ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
    FULL_VOCAB_SCORING_VERSION,
    GENERATION_VERSION,
    GREEDY_MATCH_RULE,
    answer_token_table,
    scoring_contract_digest,
)
from jlens.mmpilot.store import payload_checksum, safe_key
from jlens.mmpilot.validated_band_followup import (
    EXCLUDED_FAILED_LAYER,
    FOLLOWUP_BAND_END,
    FOLLOWUP_BAND_START,
    FOLLOWUP_PRIMARY_BAND,
    FOLLOWUP_REPORT_NAME,
    FOLLOWUP_REPORT_SCHEMA,
    FOLLOWUP_STUDY_NAME,
    FOLLOWUP_SUFFIX_STARTS,
)

__all__ = [
    "BAND_FOLLOWUP_FINGERPRINT_PIN",
    "BAND_FOLLOWUP_REPORT_CHECKSUM_PIN",
    "CANONICAL_AUDIO_GENERATION_FINGERPRINT_PIN",
    "CANONICAL_AUDIO_REPORT_CHECKSUM_PIN",
    "FULL_VOCAB_NOT_EVALUATED",
    "FULL_VOCAB_PROTOCOL_VERSION",
    "FULL_VOCAB_RUN_PREFIX",
    "FULL_VOCAB_STUDY_NAME",
    "GREEDY_SUBSET",
    "PASS_CAP",
    "REQUIRED_PINS",
    "VERDICT_NAMES",
    "FullVocabRefused",
    "FullVocabThresholds",
    "assert_prompt_reconstruction",
    "read_historical_prompt_hashes",
    "conditional_logprob_verdict",
    "cross_modal_conjunction",
    "family_a_trials",
    "format_pass_budget",
    "format_verdicts",
    "full_vocab_design_record",
    "full_vocab_fingerprint",
    "full_vocab_pass_budget",
    "full_vocab_report",
    "read_band_followup_report",
    "read_canonical_audio_provenance",
    "require_pin",
    "resolve_study_tokens",
    "reuse_completed_population",
    "summarize_full_vocab_cells",
    "trial_key",
    "unrestricted_reasoning_verdict",
]

# ------------------------------------------------------------------- versions

FULL_VOCAB_STUDY_NAME = "FULL_VOCABULARY_CAUSAL_VALIDATION"
FULL_VOCAB_PROTOCOL_VERSION = "mmpilot.full_vocabulary_causal_validation.v1"
FULL_VOCAB_DESIGN_VERSION = "mmpilot.full_vocabulary_causal_design.v1"
FULL_VOCAB_VERDICT_VERSION = "mmpilot.full_vocabulary_causal_verdict.v1"
FULL_VOCAB_REPORT_SCHEMA = "jlens.mmpilot.full_vocabulary_causal_report.v1"
FULL_VOCAB_REPORT_NAME = "full_vocabulary_causal_validation_report.json"

#: A new run prefix, distinct from every completed family, so a resume can never
#: land in one of them.
FULL_VOCAB_RUN_PREFIX = "mmfv"

#: The unit family. A completed restricted-candidate unit can never be read as
#: one of these and vice versa.
FULL_VOCAB_UNIT_FAMILY = "full_vocabulary_unrestricted_next_token"

#: The hard resource cap for this study. Exceeding it stops the run *before* a
#: model loads, with the driving factor named.
PASS_CAP = 5_000


class FullVocabRefused(RuntimeError):
    """The corrected experiment cannot be constructed as specified."""


# --------------------------------------------------------- the required pins

#: The completed L33-L40 follow-up: its run directory name, its report's own
#: ``report_checksum``, and its run fingerprint digest. Supplied out of band by
#: the operator who holds the record — never read from the run being verified.
BAND_FOLLOWUP_RUN_NAME = "band3340_real_2a72bda9b4ba"
BAND_FOLLOWUP_REPORT_CHECKSUM_PIN = (
    "sha256:f808ac89236c640269698d18c999412e0164533349b69a4d9960cdcc1ce263cb"
)
BAND_FOLLOWUP_FINGERPRINT_PIN = (
    "sha256:2a72bda9b4bad352d93e387ba7d1dd109b3e7f7d6a14093638e4c3a7ee1c412e"
)

#: The canonical three-modality run and its **raw-generation** fingerprint,
#: which is the digest recorded in ``docs/three_modality_claim_admissibility.md``.
CANONICAL_AUDIO_RUN_NAME = "mmaudio_native_audio_transfer_20260806T144822"
CANONICAL_AUDIO_GENERATION_FINGERPRINT_PIN = (
    "sha256:c868999e3d59fba5a44fda9ed5f4815c8f6085432ec902552180f70896665920"
)

#: **Deliberately empty.** The canonical run's *report* checksum is not recorded
#: anywhere in this repository, and the one number this study may not accept is
#: a checksum printed by the run it is verifying. Family B refuses until an
#: operator writes the out-of-band value here.
CANONICAL_AUDIO_REPORT_CHECKSUM_PIN = ""

#: Same for the capability-filtered v2 amendment beside it.
CANONICAL_AUDIO_AMENDED_SUMMARY_CHECKSUM_PIN = ""

#: Pin name -> (value, what it identifies). Printed by the preflight so an
#: operator can see at a glance which family is currently runnable.
REQUIRED_PINS: dict[str, dict] = {
    "BAND_FOLLOWUP_REPORT_CHECKSUM": {
        "value": BAND_FOLLOWUP_REPORT_CHECKSUM_PIN,
        "identifies": f"{FOLLOWUP_REPORT_NAME} in {BAND_FOLLOWUP_RUN_NAME}",
        "family": "A",
    },
    "BAND_FOLLOWUP_FINGERPRINT": {
        "value": BAND_FOLLOWUP_FINGERPRINT_PIN,
        "identifies": f"the run fingerprint digest of {BAND_FOLLOWUP_RUN_NAME}",
        "family": "A",
    },
    "CANONICAL_AUDIO_GENERATION_FINGERPRINT": {
        "value": CANONICAL_AUDIO_GENERATION_FINGERPRINT_PIN,
        "identifies": f"the raw-generation fingerprint of {CANONICAL_AUDIO_RUN_NAME}",
        "family": "B",
    },
    "CANONICAL_AUDIO_REPORT_CHECKSUM": {
        "value": CANONICAL_AUDIO_REPORT_CHECKSUM_PIN,
        "identifies": (
            f"native_audio_transfer_summary.json in {CANONICAL_AUDIO_RUN_NAME}"
        ),
        "family": "B",
    },
    "CANONICAL_AUDIO_AMENDED_SUMMARY_CHECKSUM": {
        "value": CANONICAL_AUDIO_AMENDED_SUMMARY_CHECKSUM_PIN,
        "identifies": (
            "native_audio_transfer_summary_capability_filtered_v2.json in "
            f"{CANONICAL_AUDIO_RUN_NAME}"
        ),
        "family": "B",
    },
}


def require_pin(name: str, *, pins: Mapping[str, Mapping] | None = None) -> str:
    """The pinned value, or a refusal that names the pin and says why.

    Raises:
        FullVocabRefused: If the pin is empty. This is the *designed* behaviour
            for a checksum nobody has recorded: an unpinned artifact would have
            to be trusted on the word of the run being verified, and that is the
            one source this study may not accept.
    """
    table = dict(pins or REQUIRED_PINS)
    if name not in table:
        raise FullVocabRefused(f"unknown provenance pin {name!r}")
    value = str(table[name].get("value") or "")
    if not value:
        raise FullVocabRefused(
            f"the required configuration pin {name} is empty. It identifies "
            f"{table[name]['identifies']}. Set it to the checksum you hold out "
            "of band and re-run. Do NOT copy a checksum printed by the run this "
            "study is verifying: a report that vouches for itself proves nothing"
        )
    return value


# ------------------------------------------------------ reading family A's run


def _read_json(path: Path, *, what: str) -> dict:
    if not path.exists():
        raise FullVocabRefused(f"no {what} at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise FullVocabRefused(f"{path} is not readable JSON: {error}") from error
    if not isinstance(payload, dict):
        raise FullVocabRefused(f"{path} is not a JSON object")
    return payload


def read_band_followup_report(
    followup_run_dir: str | os.PathLike[str],
    *,
    expected_report_checksum: str | None = None,
    expected_fingerprint: str | None = None,
    require_real_mode: bool = True,
) -> tuple[Path, dict]:
    """Read and verify the completed L33-L40 follow-up report, read-only.

    The directory is **given**, never discovered: no glob, no newest-first sort.
    Every clause is a refusal.

    Raises:
        FullVocabRefused: On a missing file, a wrong schema, a checksum that
            differs from its pin, a body that does not rehash to its own
            recorded checksum, a MOCK report, or a fingerprint mismatch.
    """
    checksum_pin = (
        expected_report_checksum
        if expected_report_checksum is not None
        else require_pin("BAND_FOLLOWUP_REPORT_CHECKSUM")
    )
    fingerprint_pin = (
        expected_fingerprint
        if expected_fingerprint is not None
        else require_pin("BAND_FOLLOWUP_FINGERPRINT")
    )
    root = Path(followup_run_dir)
    path = root / FOLLOWUP_REPORT_NAME
    report = _read_json(path, what="L33-L40 follow-up report")
    problems: list[str] = []

    if report.get("schema") != FOLLOWUP_REPORT_SCHEMA:
        problems.append(
            f"schema is {report.get('schema')!r}, not {FOLLOWUP_REPORT_SCHEMA!r}"
        )
    if report.get("study_name") != FOLLOWUP_STUDY_NAME:
        problems.append(
            f"study_name is {report.get('study_name')!r}, not {FOLLOWUP_STUDY_NAME!r}"
        )
    recorded = str(report.get("report_checksum"))
    if recorded != str(checksum_pin):
        problems.append(
            f"report_checksum is {recorded!r}, not the pinned {checksum_pin!r}"
        )
    recomputed = payload_checksum(
        {key: value for key, value in report.items() if key != "report_checksum"}
    )
    if recomputed != recorded:
        problems.append(
            f"the report body hashes to {recomputed!r} but records {recorded!r}. "
            "Report this rather than editing anything"
        )
    if require_real_mode and report.get("mode") != "real":
        problems.append(
            f"mode is {report.get('mode')!r}; a MOCK report reruns nothing"
        )
    fingerprint = dict(report.get("fingerprint") or {})
    digest = str(fingerprint.pop("followup_fingerprint_digest", None))
    recomputed_fingerprint = payload_checksum(fingerprint)
    if digest != recomputed_fingerprint:
        problems.append(
            f"the embedded follow-up configuration hashes to "
            f"{recomputed_fingerprint!r} but records {digest!r}"
        )
    run_digest = str((report.get("resume") or {}).get("fingerprint_digest"))
    if run_digest != str(fingerprint_pin):
        problems.append(
            f"resume.fingerprint_digest is {run_digest!r}, not the pinned "
            f"run fingerprint {fingerprint_pin!r}"
        )
    if problems:
        raise FullVocabRefused(
            f"{path} is not the completed L33-L40 follow-up this rerun corrects:"
            "\n  - " + "\n  - ".join(problems)
        )
    return path, report


def reuse_completed_population(report: Mapping) -> dict:
    """The exact population the completed run measured. Nothing is redrawn.

    Reads ``population_groups`` and ``capability_selection.selected_group_ids``
    out of the immutable report and refuses if either is absent. There is no
    corpus scan, no ``hidden_animal_population`` call and no capability
    reselection anywhere in this function — enlarging, shrinking or
    re-qualifying the population would make this an independent replication
    with a different population, which is precisely what it is not.

    Raises:
        FullVocabRefused: If the report carries no usable population, if the
            groups are not image/group unique, or if a selected group id is not
            in the population.
    """
    groups = [dict(row) for row in (report.get("population_groups") or ())]
    if not groups:
        raise FullVocabRefused(
            "the completed report carries no population_groups. The exact "
            "historical population cannot be reconstructed from an immutable "
            "artifact, so this family is refused rather than approximated"
        )
    selection = dict(report.get("capability_selection") or {})
    selected = {
        str(key): [str(value) for value in values]
        for key, values in dict(selection.get("selected_group_ids") or {}).items()
    }
    if not selected:
        raise FullVocabRefused(
            "the completed report records no capability_selection."
            "selected_group_ids. Which cells the causal stage actually spent "
            "passes on is part of the historical design and will not be guessed"
        )
    group_ids = [row.get("group_id") for row in groups]
    image_ids = [row.get("image_id") for row in groups]
    if len(set(group_ids)) != len(group_ids):
        raise FullVocabRefused("the completed population repeats a group id")
    if len(set(image_ids)) != len(image_ids):
        raise FullVocabRefused("the completed population repeats an image id")
    known = set(group_ids)
    missing = sorted(
        {value for values in selected.values() for value in values} - known
    )
    if missing:
        raise FullVocabRefused(
            f"selected group id(s) {missing} are not in population_groups; the "
            "selection and the population disagree"
        )
    payload = {
        "source_run": report.get("run_dir") or BAND_FOLLOWUP_RUN_NAME,
        "source_report_checksum": report.get("report_checksum"),
        "reused_verbatim": True,
        "redrawn": False,
        "enlarged": False,
        "capability_reselected": False,
        "is_a_measurement_correction_rerun_on_the_same_population": True,
        "is_an_independent_replication": False,
        "n_groups": len(groups),
        "n_distinct_images": len(set(image_ids)),
        "group_ids": [str(value) for value in group_ids],
        "image_ids": [str(value) for value in image_ids],
        "groups": groups,
        "selected_group_ids": selected,
        "original_population_digest": (report.get("population") or {}).get(
            "population_digest"
        ),
    }
    return {**payload, "population_reuse_digest": payload_checksum(
        {k: v for k, v in payload.items() if k != "groups"}
    )}


def read_canonical_audio_provenance(
    audio_run_dir: str | os.PathLike[str],
    *,
    summary_name: str = "native_audio_transfer_summary.json",
    amended_summary_name: str = (
        "native_audio_transfer_summary_capability_filtered_v2.json"
    ),
    expected_report_checksum: str | None = None,
    expected_amended_checksum: str | None = None,
    expected_generation_fingerprint: str | None = None,
) -> dict:
    """Read the canonical three-modality run's immutable artifacts, read-only.

    Family B exists only if this succeeds. It requires the *report* checksum
    pin, which is deliberately empty in this module — so on a repository where
    nobody has recorded it, family B refuses here, before any model spending,
    with a message naming the pin. That refusal is the specified behaviour, not
    a bug to work around by trusting the file.

    Raises:
        FullVocabRefused: On an empty pin, a missing artifact, a checksum
            mismatch, or a fingerprint that is not the pinned raw-generation one.
    """
    checksum_pin = (
        expected_report_checksum
        if expected_report_checksum is not None
        else require_pin("CANONICAL_AUDIO_REPORT_CHECKSUM")
    )
    amended_pin = (
        expected_amended_checksum
        if expected_amended_checksum is not None
        else require_pin("CANONICAL_AUDIO_AMENDED_SUMMARY_CHECKSUM")
    )
    fingerprint_pin = (
        expected_generation_fingerprint
        if expected_generation_fingerprint is not None
        else require_pin("CANONICAL_AUDIO_GENERATION_FINGERPRINT")
    )
    root = Path(audio_run_dir)
    problems: list[str] = []

    summary_path = root / summary_name
    summary = _read_json(summary_path, what="canonical audio summary")
    recorded = str(summary.get("report_checksum") or summary.get("summary_checksum"))
    if recorded != str(checksum_pin):
        problems.append(
            f"{summary_name} checksum is {recorded!r}, not the pinned "
            f"{checksum_pin!r}"
        )

    amended_path = root / amended_summary_name
    amended = _read_json(amended_path, what="capability-filtered v2 summary")
    amended_recorded = str(
        amended.get("report_checksum") or amended.get("summary_checksum")
    )
    if amended_recorded != str(amended_pin):
        problems.append(
            f"{amended_summary_name} checksum is {amended_recorded!r}, not the "
            f"pinned {amended_pin!r}"
        )

    fingerprint_path = root / "fingerprint.json"
    fingerprint = _read_json(fingerprint_path, what="canonical audio fingerprint")
    digest = str(fingerprint.get("digest") or fingerprint.get("fingerprint_digest"))
    if digest != str(fingerprint_pin):
        problems.append(
            f"fingerprint digest is {digest!r}, not the pinned {fingerprint_pin!r}"
        )
    if problems:
        raise FullVocabRefused(
            f"{root} is not the canonical three-modality run this rerun "
            "revalidates:\n  - " + "\n  - ".join(problems)
        )
    return {
        "run_dir": str(root),
        "run_name": CANONICAL_AUDIO_RUN_NAME,
        "summary_path": str(summary_path),
        "summary_checksum": recorded,
        "amended_summary_path": str(amended_path),
        "amended_summary_checksum": amended_recorded,
        "generation_fingerprint": digest,
        "summary": summary,
        "amended_summary": amended,
        "reused_verbatim": True,
        "reselected_from_corpus": False,
        "concepts_substituted": False,
    }


# --------------------------------------------------------- token requirements

#: The reasoning experiment's answers. ``two`` and ``four`` are **required**: if
#: either is not a single token there is no global top-1 to measure and the
#: family is refused before the model loads.
REASONING_PROPERTY_ANSWERS: tuple[str, ...] = ("two", "four")
REASONING_IDENTITY_ANSWERS: tuple[str, ...] = ("bird", "cat")
REQUIRED_SINGLE_TOKEN_ANSWERS: tuple[str, ...] = (
    *REASONING_PROPERTY_ANSWERS,
    *REASONING_IDENTITY_ANSWERS,
)


def resolve_study_tokens(
    backend,
    *,
    reasoning_answers: Sequence[str] = REQUIRED_SINGLE_TOKEN_ANSWERS,
    required: Sequence[str] = REQUIRED_SINGLE_TOKEN_ANSWERS,
    three_modality_concepts: Sequence[str] = (),
) -> dict:
    """Resolve every answer through the pinned tokenizer, at runtime.

    Family A's answers are all required to be single tokens. Family B's concepts
    are *not*: ``microwave`` may well be several tokens in this vocabulary. A
    multi-token concept is recorded as unsupported for the unrestricted
    endpoint and reported separately as a sequence-likelihood diagnostic —
    never truncated to its first token, and never swapped for a different
    concept, which would make the design depend on the tokenizer's output.

    Raises:
        MultiTokenAnswerError: If ``two``, ``four``, ``bird`` or ``cat`` is not
            a single token. That refusal happens before any model spending.
    """
    reasoning = answer_token_table(
        backend, list(reasoning_answers), required=list(required)
    )
    three_modality = (
        answer_token_table(backend, list(three_modality_concepts), required=())
        if three_modality_concepts
        else {
            "supported": {},
            "unsupported": {},
            "all_single_token": True,
            "token_ids": {},
            "required": [],
            "leading_space": True,
            "scoring_version": FULL_VOCAB_SCORING_VERSION,
        }
    )
    payload = {
        "scoring_version": FULL_VOCAB_SCORING_VERSION,
        "reasoning": reasoning,
        "three_modality": three_modality,
        "family_a_supported": reasoning["all_single_token"],
        "family_b_single_token_subset": sorted(three_modality["supported"]),
        "family_b_unsupported_for_unrestricted_endpoint": sorted(
            three_modality["unsupported"]
        ),
        "unsupported_are_reported_as": ENDPOINT_CONDITIONAL_LOGPROB,
        "first_token_truncation_used": False,
        "outcome_dependent_replacement_used": False,
    }
    return {**payload, "token_digest": payload_checksum(payload)}


# ------------------------------------------------------------------ the design

#: Family A's readouts. ``property`` (two/four) is the endpoint; ``identity``
#: (bird/cat) stays an intervention-integrity diagnostic, exactly as before.
READOUT_ARMS: tuple[str, ...] = ("identity", "property")
MODALITIES: tuple[str, ...] = ("text", "image", "spoken_audio")
DIRECTED_PAIRS: tuple[tuple[str, str], ...] = (("bird", "cat"), ("cat", "bird"))

#: The predeclared greedy-demonstration subset. Frozen here, before anything
#: runs, and bound into the fingerprint with its own pass cost.
GREEDY_SUBSET: dict = {
    "family": "A",
    "modality": "text",
    "readout": "property",
    "band": list(FOLLOWUP_PRIMARY_BAND),
    "arm": "intermediate",
    "conditions": ("clean", "swap_alpha1", "zero"),
    "max_new_tokens": 4,
    "role": "secondary behavioural demonstration; never a statistical endpoint",
}


@dataclass(frozen=True)
class FullVocabThresholds:
    """Frozen pass criteria for one cell under the unrestricted endpoint.

    Attributes:
        min_images: Independent photographs (or recordings of them) per cell.
        min_unique_global_top1_rate: The primary rate — the fraction of trials
            in which the target answer is the **unique global argmax** over the
            entire vocabulary. Tie-generous and rank-based statistics are
            recorded beside it and never substituted for it.
        control_margin: How far the primary rate must exceed each matched
            control's. Zero means strictly greater.
        max_identity_flip_rate_answer_arm: Unchanged from the completed study.
    """

    min_images: int = 4
    min_unique_global_top1_rate: float = 0.50
    control_margin: float = 0.0
    max_identity_flip_rate_answer_arm: float = 0.25
    blocking_controls: tuple[str, ...] = ("zero", "random", "unrelated")

    def __post_init__(self) -> None:
        if self.min_images < 2:
            raise FullVocabRefused("min_images must be at least 2")
        for name in ("min_unique_global_top1_rate", "max_identity_flip_rate_answer_arm"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise FullVocabRefused(f"{name} must be in [0, 1], got {value}")
        if not self.blocking_controls:
            raise FullVocabRefused("at least one blocking control is required")

    def to_dict(self) -> dict:
        return {key: _plain(value) for key, value in asdict(self).items()}

    @property
    def digest(self) -> str:
        return payload_checksum(
            {"version": FULL_VOCAB_VERDICT_VERSION, **self.to_dict()}
        )


def _plain(value):
    return list(value) if isinstance(value, tuple) else value


def full_vocab_design_record(
    *,
    suffix_starts: Sequence[int] = FOLLOWUP_SUFFIX_STARTS,
    band_end: int = FOLLOWUP_BAND_END,
    modalities: Sequence[str] = MODALITIES,
    readouts: Sequence[str] = READOUT_ARMS,
    directed_pairs: Sequence[Sequence[str]] = DIRECTED_PAIRS,
    conditions: Sequence[str] = BAND_CONDITIONS,
    arms: Sequence[str] = BAND_ARMS,
    thresholds: FullVocabThresholds | None = None,
    greedy_subset: Mapping = GREEDY_SUBSET,
) -> dict:
    """The frozen design. Family A's is the completed follow-up's, unchanged.

    Raises:
        FullVocabRefused: If a band would include the excluded layer, or if the
            suffix starts are not the completed study's.
    """
    thresholds = thresholds or FullVocabThresholds()
    starts = tuple(int(value) for value in suffix_starts)
    if starts != tuple(FOLLOWUP_SUFFIX_STARTS):
        raise FullVocabRefused(
            f"suffix starts {list(starts)} are not the completed follow-up's "
            f"{list(FOLLOWUP_SUFFIX_STARTS)}. This is a measurement correction "
            "on a frozen design; the band topology is not a free parameter"
        )
    bands = [tuple(range(start, int(band_end) + 1)) for start in starts]
    for band in bands:
        if EXCLUDED_FAILED_LAYER in band:
            raise FullVocabRefused(
                f"band {list(band)} contains the excluded layer "
                f"L{EXCLUDED_FAILED_LAYER}. It failed the frozen "
                "coverage/nondegeneracy clause and there is no configuration of "
                "this study that admits it"
            )
        if min(band) < FOLLOWUP_BAND_START:
            raise FullVocabRefused(
                f"band {list(band)} starts before L{FOLLOWUP_BAND_START}"
            )
    payload = {
        "design_version": FULL_VOCAB_DESIGN_VERSION,
        "protocol_version": FULL_VOCAB_PROTOCOL_VERSION,
        "study_name": FULL_VOCAB_STUDY_NAME,
        "primary_endpoint": ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        "primary_endpoint_definition": (
            "run the original prompt with no answer candidate appended, apply "
            "the intervention during that forward pass, inspect the complete "
            "next-token distribution at the final prompt position, and count a "
            "success only when the target-appropriate token is the unique "
            "global argmax over the entire vocabulary"
        ),
        "secondary_endpoints": [
            ENDPOINT_RESTRICTED_CANDIDATE,
            ENDPOINT_CONDITIONAL_LOGPROB,
            ENDPOINT_GENERATION,
        ],
        "scoring_version": FULL_VOCAB_SCORING_VERSION,
        "scoring_contract_digest": scoring_contract_digest(),
        "generation_version": GENERATION_VERSION,
        "greedy_match_rule": GREEDY_MATCH_RULE,
        # --- family A
        "predeclared_suffix_starts": list(starts),
        "band_end": int(band_end),
        "bands": [list(band) for band in bands],
        "band_keys": [band_key(band) for band in bands],
        "primary_band": list(FOLLOWUP_PRIMARY_BAND),
        "excluded_layer": int(EXCLUDED_FAILED_LAYER),
        "arms": [str(value) for value in arms],
        "conditions": [str(value) for value in conditions],
        "condition_alpha": {str(k): float(v) for k, v in CONDITION_ALPHA.items()},
        "primary_alpha": PRIMARY_ALPHA,
        "secondary_alpha": SECONDARY_ALPHA,
        "alpha_roles": {
            str(PRIMARY_ALPHA): "primary",
            str(SECONDARY_ALPHA): "prespecified sensitivity, never primary evidence",
        },
        "modalities": [str(value) for value in modalities],
        "readouts": [str(value) for value in readouts],
        "primary_readout": "property",
        "diagnostic_readout": "identity",
        "directed_pairs": [
            {"source": str(pair[0]), "target": str(pair[1])} for pair in directed_pairs
        ],
        "greedy_subset": {
            key: _plain(value) for key, value in dict(greedy_subset).items()
        },
        "thresholds": thresholds.to_dict(),
        "threshold_digest": thresholds.digest,
        # --- what this is, stated in the design itself
        "is_a_measurement_correction_rerun_on_the_same_population": True,
        "is_an_independent_replication": False,
        "population_redrawn": False,
        "population_enlarged": False,
        "capability_reselected": False,
        "direct_answer_arm_is_a_positive_control": True,
        "direct_answer_arm_alone_establishes_no_intermediate_reasoning": True,
        "fitting_performed": False,
        "backward_passes": 0,
    }
    return {**payload, "design_digest": payload_checksum(payload)}


# --------------------------------------------------------------- the budget


def full_vocab_pass_budget(
    *,
    # --- family A, derived from the reused population
    a_cells: int,
    a_bands: int,
    a_arms: int,
    a_conditions: int,
    a_readouts: int,
    # --- family B, derived from the canonical run's own artifacts
    b_claim_supporting_cells: int = 0,
    b_samples_per_cell: int = 0,
    b_layers: int = 0,
    b_conditions: int = 0,
    b_clean_inputs: int = 0,
    # --- the predeclared greedy subset
    greedy_trials: int = 0,
    greedy_max_new_tokens: int = 0,
    cap: int = PASS_CAP,
    seconds_per_pass_low: float = 0.9,
    seconds_per_pass_high: float = 2.2,
) -> dict:
    """Exact forward-pass counts, derived from configuration and never guessed.

    The unrestricted endpoint costs **one forward pass per trial**, not one per
    candidate: the distribution is read whole, so there is nothing to iterate
    over. That single fact is what lets a faithful rerun of a 10,752
    candidate-pass design fit under a 5,000-pass cap.

    When the total exceeds ``cap`` the payload carries ``within_cap = False``,
    the largest factor, and the smallest lossless reduction — and the caller
    stops before loading a model. No condition is ever silently dropped.
    """
    for name, value in (
        ("a_cells", a_cells),
        ("a_bands", a_bands),
        ("a_arms", a_arms),
        ("a_conditions", a_conditions),
        ("a_readouts", a_readouts),
    ):
        if int(value) < 1:
            raise FullVocabRefused(f"{name} must be at least 1, got {value}")

    a_clean = int(a_cells) * int(a_readouts)
    a_intervention = (
        int(a_cells) * int(a_bands) * int(a_arms) * int(a_conditions) * int(a_readouts)
    )
    b_clean = int(b_clean_inputs)
    b_intervention = (
        int(b_claim_supporting_cells)
        * int(b_samples_per_cell)
        * int(b_layers)
        * int(b_conditions)
    )
    greedy = int(greedy_trials) * int(greedy_max_new_tokens)
    total = a_clean + a_intervention + b_clean + b_intervention + greedy

    contributions = {
        "family_a_clean": a_clean,
        "family_a_intervention": a_intervention,
        "family_b_clean": b_clean,
        "family_b_intervention": b_intervention,
        "greedy_demonstration": greedy,
    }
    largest = max(contributions, key=lambda key: contributions[key])
    within = total <= int(cap)
    payload = {
        "protocol_version": FULL_VOCAB_PROTOCOL_VERSION,
        "endpoint": ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        "passes_per_trial": 1,
        "passes_per_trial_note": (
            "one forward pass per trial for the unrestricted endpoint; the "
            "completed studies spent one per candidate"
        ),
        "family_a": {
            "clean_unrestricted_passes": a_clean,
            "intervention_and_control_passes": a_intervention,
            "factors": {
                "cells": int(a_cells),
                "bands": int(a_bands),
                "arms": int(a_arms),
                "conditions": int(a_conditions),
                "readouts": int(a_readouts),
            },
            "total": a_clean + a_intervention,
        },
        "family_b": {
            "clean_unrestricted_passes": b_clean,
            "intervention_and_control_passes": b_intervention,
            "factors": {
                "claim_supporting_cells": int(b_claim_supporting_cells),
                "samples_per_cell": int(b_samples_per_cell),
                "layers": int(b_layers),
                "conditions": int(b_conditions),
            },
            "total": b_clean + b_intervention,
        },
        "greedy_demonstration": {
            "trials": int(greedy_trials),
            "max_new_tokens": int(greedy_max_new_tokens),
            "passes": greedy,
            "is_secondary": True,
        },
        "contributions": contributions,
        "largest_contribution": largest,
        "total": total,
        "cap": int(cap),
        "within_cap": within,
        "headroom": int(cap) - total,
        "backward_passes": 0,
        "fitting_performed": False,
        "jacobian_accumulation": False,
        "l4_hours_low": round(total * float(seconds_per_pass_low) / 3600.0, 2),
        "l4_hours_high": round(total * float(seconds_per_pass_high) / 3600.0, 2),
        "checkpoint_granularity": (
            "one atomically written, checksum-valid unit per clean trial, per "
            "intervention/control trial and per greedy trial; a disconnect "
            "loses at most the forward pass currently executing"
        ),
    }
    if not within:
        payload["excess"] = total - int(cap)
        payload["excess_driven_by"] = largest
        payload["smallest_lossless_reduction"] = _smallest_lossless_reduction(
            largest, payload
        )
        payload["no_condition_is_silently_dropped"] = True
    return {**payload, "budget_checksum": payload_checksum(payload)}


def _smallest_lossless_reduction(largest: str, payload: Mapping) -> str:
    """The cheapest change that keeps every predeclared condition."""
    if largest == "greedy_demonstration":
        return (
            "the greedy demonstration is secondary: reduce max_new_tokens or the "
            "predeclared subset. No statistical condition is affected"
        )
    if largest == "family_a_intervention":
        return (
            "the identity readout is a diagnostic, not the endpoint. Scoring it "
            "only at the primary band [33..40] instead of at all four bands "
            "keeps every primary-endpoint condition and every control intact"
        )
    if largest == "family_b_intervention":
        return (
            "family B's replication layers are secondary to its primary layer. "
            "Running the primary layer first, as its own resumable pass, keeps "
            "every control and every direction at that layer"
        )
    return (
        "reuse the clean unrestricted scores across readouts that share a prompt; "
        "no condition is dropped"
    )


def format_pass_budget(budget: Mapping) -> str:
    """The block printed before any model is loaded."""
    a = budget["family_a"]
    b = budget["family_b"]
    greedy = budget["greedy_demonstration"]
    lines = [
        "=" * 78,
        f"PASS BUDGET - {FULL_VOCAB_STUDY_NAME} (forward passes only, no fitting)",
        "=" * 78,
        f"  endpoint                       {budget['endpoint']}",
        f"  passes per trial               {budget['passes_per_trial']}  "
        "(the completed studies spent one per candidate)",
        "",
        "  FAMILY A - L33-L40 reasoning, same population, corrected endpoint",
        f"    clean unrestricted           {a['clean_unrestricted_passes']:,}",
        f"    interventions + controls     {a['intervention_and_control_passes']:,}",
        "      = " + " x ".join(f"{v} {k}" for k, v in a["factors"].items()),
        f"    subtotal                     {a['total']:,}",
        "",
        "  FAMILY B - canonical three-modality causal endpoint validation",
        f"    clean unrestricted           {b['clean_unrestricted_passes']:,}",
        f"    interventions + controls     {b['intervention_and_control_passes']:,}",
        "      = " + " x ".join(f"{v} {k}" for k, v in b["factors"].items()),
        f"    subtotal                     {b['total']:,}",
        "",
        f"  greedy demonstration           {greedy['passes']:,}  "
        f"({greedy['trials']} trials x {greedy['max_new_tokens']} new tokens, "
        "secondary)",
        "",
        f"  TOTAL                          {budget['total']:,}",
        f"  cap                            {budget['cap']:,}   "
        f"within cap: {budget['within_cap']}   headroom: {budget['headroom']:,}",
        f"  backward passes                {budget['backward_passes']}  "
        f"(fitting_performed={budget['fitting_performed']}, "
        f"jacobian_accumulation={budget['jacobian_accumulation']})",
        f"  L4 wall time                   {budget['l4_hours_low']:.2f}-"
        f"{budget['l4_hours_high']:.2f} h; an A100 is usually faster",
        f"  checkpoint                     {budget['checkpoint_granularity']}",
    ]
    if not budget["within_cap"]:
        lines += [
            "",
            "  " + "!" * 70,
            f"  OVER CAP BY {budget['excess']:,} PASSES. Stopping before model load.",
            f"  driven by: {budget['excess_driven_by']} "
            f"({budget['contributions'][budget['excess_driven_by']]:,} passes)",
            f"  smallest lossless reduction: {budget['smallest_lossless_reduction']}",
            "  No condition is silently dropped. Choose the reduction explicitly.",
            "  " + "!" * 70,
        ]
    return "\n".join(lines)


# --------------------------------------------------------------- fingerprint


def full_vocab_fingerprint(
    *,
    design: Mapping,
    endpoint_audit_digest: str,
    band_followup_report_checksum: str,
    band_followup_fingerprint: str,
    canonical_audio_provenance: Mapping | None,
    population_reuse: Mapping,
    lens_checksums: Mapping[int, str],
    media_checksums: Mapping | None,
    model_repo_id: str,
    model_revision: str,
    processor_revision: str,
    transformers_version: str,
    audio_protocol_fingerprint: str | None,
    prompt_protocol: Mapping | Sequence[Mapping] | None,
    tokens: Mapping,
    coordinate_swap_method_version: str,
    thresholds: Mapping,
    seeds: Mapping,
    output_head_convention: str,
) -> dict:
    """Everything a corrected result is bound to. A change refuses a resume.

    Raises:
        FullVocabRefused: If a band layer has no recorded lens checksum, or if
            the endpoint-audit digest is missing. Both are load-bearing: the
            first names the artifact that defined the exchanged coordinates, the
            second names the audit that decided what the endpoint means.
    """
    if not str(endpoint_audit_digest or ""):
        raise FullVocabRefused(
            "no endpoint-audit digest. The corrected run is bound to the audit "
            "that defined its endpoint; a run without one cannot say which "
            "question it answered"
        )
    checksums = {int(layer): str(value) for layer, value in lens_checksums.items()}
    missing = sorted(
        {
            layer
            for band in design["bands"]
            for layer in band
            if int(layer) not in checksums
        }
    )
    if missing:
        raise FullVocabRefused(
            f"no lens checksum recorded for band layer(s) {missing}; every "
            "patched layer must name the validated artifact that defined its "
            "coordinates"
        )
    payload = {
        "study_family": FULL_VOCAB_UNIT_FAMILY,
        "study_name": FULL_VOCAB_STUDY_NAME,
        "protocol_version": FULL_VOCAB_PROTOCOL_VERSION,
        "design_digest": design["design_digest"],
        "full_vocabulary_scoring_version": FULL_VOCAB_SCORING_VERSION,
        "scoring_contract_digest": design["scoring_contract_digest"],
        "output_head_convention": str(output_head_convention),
        "generation_configuration": design["greedy_subset"],
        "greedy_match_rule": GREEDY_MATCH_RULE,
        # --- provenance
        "endpoint_audit_digest": str(endpoint_audit_digest),
        "band_followup_report_checksum": str(band_followup_report_checksum),
        "band_followup_fingerprint": str(band_followup_fingerprint),
        "canonical_audio_summary_checksum": (
            None
            if not canonical_audio_provenance
            else canonical_audio_provenance.get("summary_checksum")
        ),
        "canonical_audio_amended_summary_checksum": (
            None
            if not canonical_audio_provenance
            else canonical_audio_provenance.get("amended_summary_checksum")
        ),
        "canonical_audio_generation_fingerprint": (
            None
            if not canonical_audio_provenance
            else canonical_audio_provenance.get("generation_fingerprint")
        ),
        "family_b_evaluated": bool(canonical_audio_provenance),
        # --- the population, exactly
        "population_reuse_digest": population_reuse["population_reuse_digest"],
        "population_group_ids": list(population_reuse["group_ids"]),
        "population_image_ids": list(population_reuse["image_ids"]),
        "selected_group_ids": {
            str(key): list(values)
            for key, values in sorted(
                dict(population_reuse["selected_group_ids"]).items()
            )
        },
        "media_checksums": dict(media_checksums or {}),
        # --- the design
        "bands": [list(band) for band in design["bands"]],
        "band_keys": list(design["band_keys"]),
        "excluded_layer": design["excluded_layer"],
        "arms": list(design["arms"]),
        "conditions": list(design["conditions"]),
        "condition_alpha": dict(design["condition_alpha"]),
        "modalities": list(design["modalities"]),
        "readouts": list(design["readouts"]),
        "directed_pairs": [dict(pair) for pair in design["directed_pairs"]],
        "coordinate_swap_method_version": str(coordinate_swap_method_version),
        "artifact_checksums": {
            str(layer): checksums[layer] for layer in sorted(checksums)
        },
        # --- model and protocol pins
        "model_repo_id": str(model_repo_id),
        "model_revision": str(model_revision),
        "processor_revision": str(processor_revision),
        "transformers_version": str(transformers_version),
        "audio_protocol_fingerprint": audio_protocol_fingerprint,
        "prompt_protocol": (
            [dict(item) for item in prompt_protocol]
            if isinstance(prompt_protocol, (list, tuple))
            else (dict(prompt_protocol) if prompt_protocol else None)
        ),
        "token_digest": tokens["token_digest"],
        "answer_token_ids": dict(tokens["reasoning"]["token_ids"]),
        # --- how it is judged
        "thresholds": dict(thresholds),
        "threshold_digest": design["threshold_digest"],
        "seeds": dict(seeds),
        "verdict_version": FULL_VOCAB_VERDICT_VERSION,
        "no_fitting_entry_point_is_reachable": True,
        "backward_passes": 0,
    }
    return {**payload, "full_vocab_fingerprint_digest": payload_checksum(payload)}


# ------------------------------------------------------------ trial planning


def trial_key(
    *,
    family: str,
    group_id: str,
    modality: str,
    readout: str,
    band_or_layer: str,
    arm: str,
    condition: str,
    kind: str = "trial",
) -> str:
    """Stable per-trial unit key. Also the resume key."""
    return safe_key(
        f"fv-{family}",
        kind,
        group_id,
        modality,
        readout,
        band_or_layer,
        arm,
        condition,
    )


def family_a_trials(
    population_reuse: Mapping, design: Mapping
) -> list[dict]:
    """Every family-A trial the design calls for, in a deterministic order.

    Enumerated from the reused population and the frozen design, so the trial
    list is a function of the fingerprint and nothing else. Clean trials come
    first: an intervened trial is only meaningful beside the clean one for the
    same input, and a resume that lost the clean unit must not score the edit.
    """
    by_id = {str(row["group_id"]): dict(row) for row in population_reuse["groups"]}
    selected = dict(population_reuse["selected_group_ids"])
    pairs = {
        str(pair["source"]): str(pair["target"]) for pair in design["directed_pairs"]
    }
    trials: list[dict] = []
    for cell_key in sorted(selected):
        source, modality = cell_key.split("|", 1)
        if modality not in design["modalities"]:
            continue
        target = pairs.get(source)
        if target is None:
            raise FullVocabRefused(
                f"the completed selection names source concept {source!r}, which "
                "is not in the frozen directed pairs. The population and the "
                "design disagree; nothing is substituted"
            )
        for group_id in sorted(str(value) for value in selected[cell_key]):
            group = by_id[group_id]
            for readout in design["readouts"]:
                trials.append(
                    {
                        "family": "A",
                        "kind": "clean",
                        "group_id": group_id,
                        "image_id": str(group.get("image_id")),
                        "modality": modality,
                        "readout": str(readout),
                        "source": source,
                        "target": target,
                        "band": None,
                        "arm": None,
                        "condition": "clean",
                        "alpha": None,
                        "key": trial_key(
                            family="A",
                            group_id=group_id,
                            modality=modality,
                            readout=str(readout),
                            band_or_layer="none",
                            arm="none",
                            condition="clean",
                            kind="clean",
                        ),
                    }
                )
    for cell_key in sorted(selected):
        source, modality = cell_key.split("|", 1)
        if modality not in design["modalities"]:
            continue
        target = pairs[source]
        for group_id in sorted(str(value) for value in selected[cell_key]):
            group = by_id[group_id]
            for band in design["bands"]:
                key_band = band_key(band)
                for arm in design["arms"]:
                    for condition in design["conditions"]:
                        for readout in design["readouts"]:
                            trials.append(
                                {
                                    "family": "A",
                                    "kind": "intervention",
                                    "group_id": group_id,
                                    "image_id": str(group.get("image_id")),
                                    "modality": modality,
                                    "readout": str(readout),
                                    "source": source,
                                    "target": target,
                                    "band": list(band),
                                    "band_key": key_band,
                                    "arm": str(arm),
                                    "condition": str(condition),
                                    "alpha": float(
                                        design["condition_alpha"][str(condition)]
                                    ),
                                    "key": trial_key(
                                        family="A",
                                        group_id=group_id,
                                        modality=modality,
                                        readout=str(readout),
                                        band_or_layer=key_band,
                                        arm=str(arm),
                                        condition=str(condition),
                                    ),
                                }
                            )
    if len({row["key"] for row in trials}) != len(trials):
        raise FullVocabRefused("the family-A trial plan has a duplicate unit key")
    return trials


# ------------------------------------------------- proving the inputs are the same


def read_historical_prompt_hashes(
    run_dir: str | os.PathLike[str], *, stage: str = "intervention"
) -> dict:
    """``(group_id, modality, readout) -> prompt_hash`` from a completed run.

    The completed report records which photographs and recordings were used but
    not the caption text, so the rerun rebuilds each prompt from the pinned
    manifest and then **proves** it rebuilt the historical one by comparing
    hashes. Every unit is validated against its own recorded checksum first.

    Raises:
        FullVocabRefused: If the directory holds no valid units carrying a
            prompt hash. A rerun that cannot prove it is scoring the same prompt
            is not a measurement correction.
    """
    root = Path(run_dir) / "units" / stage
    if not root.exists():
        raise FullVocabRefused(
            f"no {stage} units at {root}. The historical prompts cannot be "
            "reconstructed from an immutable artifact, so this family is "
            "refused rather than approximated"
        )
    hashes: dict[str, str] = {}
    invalid = 0
    for path in sorted(root.glob("*.json")):
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            payload = stored["payload"]
            if stored.get("unit_checksum") != payload_checksum(payload):
                invalid += 1
                continue
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            invalid += 1
            continue
        prompt_hash = payload.get("prompt_hash")
        if not prompt_hash:
            continue
        key = "|".join(
            (
                str(payload.get("group_id")),
                str(payload.get("modality")),
                str(payload.get("readout")),
            )
        )
        previous = hashes.setdefault(key, str(prompt_hash))
        if previous != str(prompt_hash):
            raise FullVocabRefused(
                f"the completed run records two different prompt hashes for "
                f"{key}; the historical input is ambiguous and will not be guessed"
            )
    if not hashes:
        raise FullVocabRefused(
            f"{root} holds no unit carrying a prompt hash; the historical "
            "prompts cannot be proven and this family is refused"
        )
    return {
        "run_dir": str(run_dir),
        "stage": stage,
        "n_prompt_hashes": len(hashes),
        "n_invalid_units": invalid,
        "prompt_hashes": hashes,
    }


def assert_prompt_reconstruction(
    *,
    group_id: str,
    modality: str,
    readout: str,
    rebuilt_prompt_hash: str,
    historical: Mapping,
) -> dict:
    """Prove one rebuilt prompt is byte-identical to the historical one.

    Raises:
        FullVocabRefused: If the key is absent or the hashes differ. An
            approximate reconstruction is refused: a rerun on a *similar* prompt
            measures a different thing and would be reported as if it did not.
    """
    key = f"{group_id}|{modality}|{readout}"
    hashes = dict(historical.get("prompt_hashes") or {})
    if key not in hashes:
        raise FullVocabRefused(
            f"the completed run records no prompt hash for {key}. The exact "
            "historical input cannot be reconstructed, so this cell is refused "
            "rather than approximated"
        )
    if str(hashes[key]) != str(rebuilt_prompt_hash):
        raise FullVocabRefused(
            f"the rebuilt prompt for {key} hashes to {rebuilt_prompt_hash!r}, "
            f"the completed run recorded {hashes[key]!r}. This is a different "
            "prompt; refusing to call the result a measurement correction"
        )
    return {
        "key": key,
        "prompt_hash": str(rebuilt_prompt_hash),
        "matches_completed_run": True,
    }


# ------------------------------------------------------------- aggregation


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_full_vocab_cells(
    records: Sequence[Mapping], *, thresholds: FullVocabThresholds | None = None
) -> list[dict]:
    """Image-level aggregation, one row per (band, arm, condition, modality, pair, readout).

    Every rate is reported at once and never collapsed: the unique-global-top1
    rate is the primary, the tie-generous rate and the median global rank are
    secondary, and the restricted-candidate rate is a labelled diagnostic that
    no clause reads.
    """
    thresholds = thresholds or FullVocabThresholds()
    cells: dict[tuple, list[Mapping]] = {}
    for row in records:
        if row.get("kind") == "clean" or row.get("trial_kind") == "clean":
            continue
        key = (
            str(row.get("band_key") or band_key(row.get("band") or [0])),
            str(row.get("arm")),
            str(row.get("condition")),
            str(row.get("modality")),
            f"{row.get('source_concept') or row.get('source_answer')}"
            f"->{row.get('target_concept') or row.get('target_answer')}",
            str(row.get("readout")),
        )
        cells.setdefault(key, []).append(row)

    out: list[dict] = []
    for key, rows in sorted(cells.items()):
        band_id, arm, condition, modality, pair, readout = key
        image_ids = sorted({str(row.get("image_id")) for row in rows})
        unique = [1.0 if row.get("target_is_unique_global_top1") else 0.0 for row in rows]
        tied = [1.0 if row.get("target_is_tied_global_top1") else 0.0 for row in rows]
        restricted = [
            1.0 if row.get("target_is_restricted_candidate_top1") else 0.0
            for row in rows
            if row.get("target_is_restricted_candidate_top1") is not None
        ]
        ranks = sorted(int(row["target_rank"]) for row in rows)
        logprob_changes = [
            float(row["target_logprob_change"])
            for row in rows
            if row.get("target_logprob_change") is not None
        ]
        out.append(
            {
                "band_key": band_id,
                "arm": arm,
                "condition": condition,
                "modality": modality,
                "pair": pair,
                "readout": readout,
                "n_trials": len(rows),
                "n_distinct_images": len(image_ids),
                "image_ids": image_ids,
                "meets_image_floor": len(image_ids) >= thresholds.min_images,
                # --- primary
                "unique_global_top1_rate": _mean(unique),
                "unique_global_top1_image_ids": sorted(
                    {
                        str(row.get("image_id"))
                        for row in rows
                        if row.get("target_is_unique_global_top1")
                    }
                ),
                # --- secondary, reported beside the primary and never instead
                "tied_global_top1_rate": _mean(tied),
                "median_target_global_rank": (
                    ranks[len(ranks) // 2] if ranks else None
                ),
                "best_target_global_rank": ranks[0] if ranks else None,
                "mean_target_logprob_change": _mean(logprob_changes),
                # --- diagnostic
                "restricted_candidate_top1_rate": _mean(restricted),
                "restricted_is_a_diagnostic_only": True,
                "endpoint": ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
            }
        )
    return out


def _cell(cells: Sequence[Mapping], **match) -> Mapping | None:
    for row in cells:
        if all(str(row.get(key)) == str(value) for key, value in match.items()):
            return row
    return None


# ------------------------------------------------------------------ verdicts

FULL_VOCAB_REASONING_ALPHA1_GO = "FULL_VOCAB_REASONING_ALPHA1_GO"
FULL_VOCAB_REASONING_ALPHA2_ONLY = "FULL_VOCAB_REASONING_ALPHA2_ONLY"
FULL_VOCAB_REASONING_NO_GO = "FULL_VOCAB_REASONING_NO_GO"
FULL_VOCAB_THREE_MODALITY_CAUSAL_GO = "FULL_VOCAB_THREE_MODALITY_CAUSAL_GO"
CONDITIONAL_LOGPROB_ONLY = "CONDITIONAL_LOGPROB_ONLY"
CAPABILITY_NO_GO = "CAPABILITY_NO_GO"
NOT_EVALUATED = "NOT_EVALUATED"
INCONCLUSIVE = "INCONCLUSIVE"

#: Every terminal name, and no others.
VERDICT_NAMES: tuple[str, ...] = (
    FULL_VOCAB_REASONING_ALPHA1_GO,
    FULL_VOCAB_REASONING_ALPHA2_ONLY,
    FULL_VOCAB_REASONING_NO_GO,
    FULL_VOCAB_THREE_MODALITY_CAUSAL_GO,
    CONDITIONAL_LOGPROB_ONLY,
    CAPABILITY_NO_GO,
    NOT_EVALUATED,
    INCONCLUSIVE,
)

#: The label a historical restricted-candidate result carries until the
#: corrected run produces an unrestricted one.
FULL_VOCAB_NOT_EVALUATED = "FULL_VOCABULARY_NOT_EVALUATED"
RESTRICTED_CANDIDATE_PREFERENCE_GO = "RESTRICTED_CANDIDATE_PREFERENCE_GO"
CONTROLLED_TARGET_LOGPROB_EFFECT = "CONTROLLED_TARGET_LOGPROB_EFFECT"


def _condition_passes(
    cells: Sequence[Mapping],
    *,
    band: str,
    arm: str,
    condition: str,
    modality: str,
    pair: str,
    thresholds: FullVocabThresholds,
) -> dict:
    """One directed cell's primary clause, with every blocking control named."""
    primary = _cell(
        cells,
        band_key=band,
        arm=arm,
        condition=condition,
        modality=modality,
        pair=pair,
        readout="property",
    )
    if primary is None:
        return {
            "band_key": band,
            "arm": arm,
            "condition": condition,
            "modality": modality,
            "pair": pair,
            "evaluated": False,
            "passed": False,
            "why": "no trials in this cell",
        }
    alpha = float(CONDITION_ALPHA.get(condition, 1.0))
    control_conditions = {
        "zero": "zero",
        "random": f"random_alpha{int(alpha)}" if alpha else "zero",
        "unrelated": f"unrelated_alpha{int(alpha)}" if alpha else "zero",
    }
    rate = float(primary["unique_global_top1_rate"] or 0.0)
    clauses = {
        "meets_image_floor": bool(primary["meets_image_floor"]),
        "unique_global_top1_rate": rate >= thresholds.min_unique_global_top1_rate,
    }
    controls = {}
    for name in thresholds.blocking_controls:
        control_condition = control_conditions[name]
        control_cell = _cell(
            cells,
            band_key=band,
            arm=arm,
            condition=control_condition,
            modality=modality,
            pair=pair,
            readout="property",
        )
        control_rate = (
            None
            if control_cell is None
            else float(control_cell["unique_global_top1_rate"] or 0.0)
        )
        beats = control_cell is not None and rate > control_rate + thresholds.control_margin
        clauses[f"beats_{name}_control"] = bool(beats)
        controls[name] = {
            "condition": control_condition,
            "alpha_matched": alpha,
            "rate": control_rate,
            "beaten": bool(beats),
        }
    return {
        "band_key": band,
        "arm": arm,
        "condition": condition,
        "alpha": alpha,
        "modality": modality,
        "pair": pair,
        "readout": "property",
        "evaluated": True,
        "unique_global_top1_rate": rate,
        "tied_global_top1_rate": primary["tied_global_top1_rate"],
        "median_target_global_rank": primary["median_target_global_rank"],
        "restricted_candidate_top1_rate": primary["restricted_candidate_top1_rate"],
        "n_distinct_images": primary["n_distinct_images"],
        "clauses": clauses,
        "controls": controls,
        "passed": all(clauses.values()),
    }


def unrestricted_reasoning_verdict(
    cells: Sequence[Mapping] | None,
    *,
    bands: Sequence[str],
    modalities: Sequence[str],
    directed_pairs: Sequence[Mapping],
    thresholds: FullVocabThresholds | None = None,
    capability_sufficient: bool = True,
    causal_stage_ran: bool = True,
) -> dict:
    """Family A's verdict, on the unrestricted endpoint only.

    α=1 is the primary. α=2 can produce
    :data:`FULL_VOCAB_REASONING_ALPHA2_ONLY` and nothing stronger, ever.
    Every direction and every modality is reported before any pooled summary,
    and the direct-answer (``answer``) arm is a positive control that cannot by
    itself establish intermediate reasoning.
    """
    thresholds = thresholds or FullVocabThresholds()
    if not causal_stage_ran:
        return _verdict_payload(
            NOT_EVALUATED,
            why="the causal stage did not run in this session",
            thresholds=thresholds,
        )
    if not capability_sufficient:
        return _verdict_payload(
            CAPABILITY_NO_GO,
            why=(
                "the clean unrestricted screen did not produce enough eligible "
                "cells. No causal trial ran, so this is not a null result"
            ),
            thresholds=thresholds,
        )
    rows = list(cells or ())
    per_cell: list[dict] = []
    for condition in ("swap_alpha1", "swap_alpha2"):
        for band in bands:
            for arm in ("intermediate", "answer"):
                for modality in modalities:
                    for pair in directed_pairs:
                        key = f"{pair['source']}->{pair['target']}"
                        per_cell.append(
                            _condition_passes(
                                rows,
                                band=str(band),
                                arm=arm,
                                condition=condition,
                                modality=str(modality),
                                pair=key,
                                thresholds=thresholds,
                            )
                        )

    def passing(condition: str, arm: str, modality: str | None = None) -> list[dict]:
        return [
            row
            for row in per_cell
            if row["condition"] == condition
            and row["arm"] == arm
            and row.get("passed")
            and (modality is None or row["modality"] == modality)
        ]

    alpha1_text = passing("swap_alpha1", "intermediate", "text")
    alpha2_text = passing("swap_alpha2", "intermediate", "text")
    if alpha1_text:
        verdict = FULL_VOCAB_REASONING_ALPHA1_GO
        why = (
            f"{len(alpha1_text)} text intermediate-arm cell(s) at alpha=1 made "
            "the target answer the unique global next-token argmax more often "
            "than every alpha-matched control did"
        )
    elif alpha2_text:
        verdict = FULL_VOCAB_REASONING_ALPHA2_ONLY
        why = (
            "no alpha=1 cell passed; "
            f"{len(alpha2_text)} did at alpha=2. Alpha=2 is a prespecified "
            "sensitivity condition and is never primary evidence"
        )
    elif any(row["evaluated"] for row in per_cell):
        verdict = FULL_VOCAB_REASONING_NO_GO
        why = (
            "no intermediate-arm cell made the target answer the unique global "
            "argmax at a rate exceeding its matched controls"
        )
    else:
        verdict = INCONCLUSIVE
        why = "the design's cells produced no evaluable trials"
    payload = _verdict_payload(verdict, why=why, thresholds=thresholds)
    payload.update(
        {
            "per_cell": per_cell,
            "per_direction": [
                {
                    "pair": f"{pair['source']}->{pair['target']}",
                    "alpha1_passing_bands": sorted(
                        {
                            row["band_key"]
                            for row in passing("swap_alpha1", "intermediate")
                            if row["pair"] == f"{pair['source']}->{pair['target']}"
                        }
                    ),
                    "alpha2_passing_bands": sorted(
                        {
                            row["band_key"]
                            for row in passing("swap_alpha2", "intermediate")
                            if row["pair"] == f"{pair['source']}->{pair['target']}"
                        }
                    ),
                }
                for pair in directed_pairs
            ],
            "per_modality": {
                str(modality): {
                    "alpha1_passing_bands": sorted(
                        {
                            row["band_key"]
                            for row in passing("swap_alpha1", "intermediate", str(modality))
                        }
                    ),
                    "alpha2_passing_bands": sorted(
                        {
                            row["band_key"]
                            for row in passing("swap_alpha2", "intermediate", str(modality))
                        }
                    ),
                }
                for modality in modalities
            },
            "direct_answer_arm": {
                "role": "positive control",
                "alpha1_passing_bands": sorted(
                    {row["band_key"] for row in passing("swap_alpha1", "answer")}
                ),
                "cannot_establish_intermediate_reasoning_alone": True,
            },
            "tri_modal_conjunction_is_our_extension": True,
            "pooled_summary_follows_per_direction_reporting": True,
        }
    )
    return {**payload, "verdict_digest": payload_checksum(payload)}


def _verdict_payload(
    verdict: str, *, why: str, thresholds: FullVocabThresholds
) -> dict:
    if verdict not in VERDICT_NAMES:
        raise FullVocabRefused(f"{verdict!r} is not a declared verdict name")
    return {
        "version": FULL_VOCAB_VERDICT_VERSION,
        "study_name": FULL_VOCAB_STUDY_NAME,
        "verdict": verdict,
        "why": why,
        "endpoint": ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        "primary_alpha": PRIMARY_ALPHA,
        "secondary_alpha": SECONDARY_ALPHA,
        "alpha2_is_never_primary_evidence": True,
        "restricted_candidate_order_alone_is_never_a_go": True,
        "positive_margin_without_global_rank_1_is_never_a_go": True,
        "greedy_text_alone_is_never_a_go": True,
        "direct_answer_arm_alone_is_never_a_go": True,
        "one_modality_never_stands_in_for_all_three": True,
        "pooled_directions_never_substitute_for_a_single_working_direction": True,
        "thresholds": thresholds.to_dict(),
        "threshold_digest": thresholds.digest,
    }


def conditional_logprob_verdict(
    cells: Sequence[Mapping] | None,
    *,
    evaluated: bool,
    replicated_cells: Sequence[Mapping] = (),
    why: str = "",
) -> dict:
    """Family B's **conditional-log-probability** replication, kept separate.

    This verdict can never be promoted into an unrestricted-output verdict and
    can never be overwritten by one. They answer different questions and are
    stored under different keys.
    """
    if not evaluated:
        verdict = NOT_EVALUATED
        why = why or (
            "family B did not run: its historical provenance could not be "
            "verified from immutable artifacts, so it was refused rather than "
            "approximated"
        )
    elif replicated_cells:
        verdict = CONDITIONAL_LOGPROB_ONLY
        why = why or (
            f"{len(replicated_cells)} historically claim-supporting cell(s) "
            "reproduced their controlled conditional-log-probability effect. "
            "This is a likelihood effect, not autonomous output"
        )
    else:
        verdict = INCONCLUSIVE
        why = why or "no historically claim-supporting cell reproduced its effect"
    payload = {
        "version": FULL_VOCAB_VERDICT_VERSION,
        "verdict": verdict,
        "why": why,
        "endpoint": ENDPOINT_CONDITIONAL_LOGPROB,
        "is_not_an_unrestricted_output_verdict": True,
        "cannot_overwrite_the_full_vocabulary_verdict": True,
        "cells": [dict(row) for row in (cells or ())],
        "replicated_cells": [dict(row) for row in replicated_cells],
    }
    return {**payload, "verdict_digest": payload_checksum(payload)}


def cross_modal_conjunction(
    unrestricted: Mapping,
    conditional: Mapping,
    *,
    modalities: Sequence[str] = MODALITIES,
) -> dict:
    """The three-modality conjunction, computed from the two separate verdicts.

    A conjunction is only claimed when the unrestricted endpoint passed in every
    modality. One modality standing in for three is refused by construction: the
    per-modality table is read directly and its emptiness is visible.
    """
    per_modality = dict(unrestricted.get("per_modality") or {})
    passing = sorted(
        str(modality)
        for modality in modalities
        if per_modality.get(str(modality), {}).get("alpha1_passing_bands")
    )
    complete = len(passing) == len(tuple(modalities))
    verdict = (
        FULL_VOCAB_THREE_MODALITY_CAUSAL_GO
        if complete and unrestricted.get("verdict") == FULL_VOCAB_REASONING_ALPHA1_GO
        else NOT_EVALUATED
        if unrestricted.get("verdict") == NOT_EVALUATED
        else INCONCLUSIVE
    )
    payload = {
        "version": FULL_VOCAB_VERDICT_VERSION,
        "verdict": verdict,
        "endpoint": ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        "modalities_required": [str(value) for value in modalities],
        "modalities_passing_alpha1": passing,
        "conjunction_complete": complete,
        "unrestricted_verdict": unrestricted.get("verdict"),
        "conditional_logprob_verdict": conditional.get("verdict"),
        "the_two_verdicts_are_separate_and_neither_substitutes_for_the_other": True,
        "tri_modal_conjunction_is_our_extension": True,
    }
    return {**payload, "verdict_digest": payload_checksum(payload)}


def format_verdicts(
    unrestricted: Mapping,
    conditional: Mapping,
    conjunction: Mapping,
) -> str:
    """The three verdicts, printed separately and never merged."""
    lines = [
        "=" * 78,
        f"{FULL_VOCAB_STUDY_NAME}",
        "=" * 78,
        "",
        "  1. UNRESTRICTED OUTPUT (primary)",
        f"     verdict   {unrestricted.get('verdict')}",
        f"     why       {unrestricted.get('why')}",
    ]
    for row in unrestricted.get("per_direction") or ():
        lines.append(
            f"       {row['pair']:<14} alpha1 bands={row['alpha1_passing_bands']}  "
            f"alpha2 bands={row['alpha2_passing_bands']}"
        )
    for modality, row in (unrestricted.get("per_modality") or {}).items():
        lines.append(
            f"       {modality:<14} alpha1 bands={row['alpha1_passing_bands']}"
        )
    lines += [
        "",
        "  2. CONDITIONAL LOG-PROBABILITY (separate; never a substitute)",
        f"     verdict   {conditional.get('verdict')}",
        f"     why       {conditional.get('why')}",
        "",
        "  3. CROSS-MODAL CONJUNCTION (our extension)",
        f"     verdict   {conjunction.get('verdict')}",
        f"     passing   {conjunction.get('modalities_passing_alpha1')}",
        "",
        "  alpha=2 is a prespecified sensitivity condition and is never primary.",
        "  A restricted-candidate preference is never a full-vocabulary GO.",
        "  The direct-answer arm is a positive control, not evidence of",
        "  intermediate reasoning.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------- report


def full_vocab_report(
    *,
    mode: str,
    design: Mapping,
    fingerprint: Mapping | None,
    endpoint_audit: Mapping,
    population_reuse: Mapping,
    band_followup_provenance: Mapping,
    canonical_audio_provenance: Mapping | None,
    tokens: Mapping,
    budget: Mapping,
    cells: Sequence[Mapping],
    unrestricted: Mapping,
    conditional: Mapping,
    conjunction: Mapping,
    greedy: Sequence[Mapping] = (),
    hook_integrity: Mapping | None = None,
    fitting_audit: Mapping | None = None,
    resume: Mapping | None = None,
    run_dir: str | None = None,
) -> dict:
    """The corrected run's report, including everything it must say about itself."""
    payload = {
        "schema": FULL_VOCAB_REPORT_SCHEMA,
        "mode": str(mode),
        "study_name": FULL_VOCAB_STUDY_NAME,
        "protocol_version": FULL_VOCAB_PROTOCOL_VERSION,
        "study_family": FULL_VOCAB_UNIT_FAMILY,
        "run_dir": run_dir,
        "primary_endpoint": ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        "primary_endpoint_definition": design["primary_endpoint_definition"],
        "design": dict(design),
        "fingerprint": dict(fingerprint or {}),
        "endpoint_audit_digest": endpoint_audit.get("audit_digest"),
        "claim_ledger_digest": endpoint_audit.get("claim_ledger_digest"),
        "population_reuse": {
            key: value
            for key, value in dict(population_reuse).items()
            if key != "groups"
        },
        "band_followup_provenance": dict(band_followup_provenance),
        "canonical_audio_provenance": (
            {
                key: value
                for key, value in dict(canonical_audio_provenance).items()
                if key not in ("summary", "amended_summary")
            }
            if canonical_audio_provenance
            else None
        ),
        "family_b_evaluated": bool(canonical_audio_provenance),
        "token_requirements": dict(tokens),
        "budget": dict(budget),
        "cells": [dict(row) for row in cells],
        # --- three verdicts, three keys, never merged
        "unrestricted_output_verdict": dict(unrestricted),
        "conditional_logprob_verdict": dict(conditional),
        "cross_modal_conjunction": dict(conjunction),
        "greedy_demonstrations": [dict(row) for row in greedy],
        "greedy_is_secondary": True,
        "hook_integrity": dict(hook_integrity or {}),
        "fitting_audit": dict(fitting_audit or {}),
        "resume": dict(resume or {}),
        # --- the reporting boundary, in the report itself
        "is_a_measurement_correction_rerun_on_the_same_population": True,
        "is_an_independent_replication": False,
        "population_redrawn": False,
        "capability_reselected": False,
        "excluded_layer": int(EXCLUDED_FAILED_LAYER),
        "completed_reports_modified": False,
        "completed_units_modified": False,
        "scientific_recompute_of_completed_runs": 0,
        "alpha2_is_sensitivity_not_primary_evidence": True,
        "restricted_candidate_scoring_is_a_labelled_secondary_diagnostic": True,
        "spokencoco_tests_linguistic_spoken_captions_not_environmental_sound": True,
        "model_outputs_text_image_and_audio_are_input_modalities": True,
        "mock_proves_pipeline_only": str(mode) != "real",
        "public_method_reference": (
            "https://transformer-circuits.pub/2026/workspace/index.html"
            "#technical-details-of-j-lens-use-cases"
        ),
    }
    payload["report_checksum"] = payload_checksum(payload)
    return payload
