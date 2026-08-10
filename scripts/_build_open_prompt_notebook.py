# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Generate ``notebooks/archive/protocol_mocks/open_prompt_protocol_mock_colab.ipynb``.

The notebook is written from source here rather than edited inside a JSON blob,
so the committed file stays output-free and byte-reproducible.

Run with ``python scripts/_build_open_prompt_notebook.py`` after changing a cell.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "notebooks" / "archive" / "protocol_mocks" / "open_prompt_protocol_mock_colab.ipynb"

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


markdown(
    """
# The open prompt protocol — MOCK only

The completed three-modality causal study asked its behavioural question by
**listing every candidate in the prompt**:

```
Question: which one of these is present: bird, cat, giraffe, microwave,
toilet, zebra? Answer with exactly one word.
Answer:
```

That question is still exactly where it was, it still produces the same bytes,
and the result it produced stands — as a **candidate-conditioned** identification
result. Every candidate concept, including what a future study would use as a
swap *target*, was introduced into the model's own input.

Anthropic's strongest Global Workspace interventions do not ask that way. *"The
number of legs on the animal that spins webs is ..."* never writes `spider` and
never writes `ant`. This notebook exercises the protocol that lets the planned
coordinate-swap study ask the same way: the neutral question and the scored
candidate answers are **separate objects**, and only the question is ever built
into a prompt.

## The five protocols, and the domain each one presumes

| identifier | domain | candidates in prompt | source may appear | target may appear | supports |
|---|---|---|---|---|---|
| `mmpilot.candidate_listed_identification.v1` | — | yes (legacy) | yes | yes | candidate-conditioned identification |
| `mmpilot.open_animal_identification.v1` | `animal` | no | yes, in natural evidence, recorded | never | open cross-modal **animal** identification |
| `mmpilot.open_entity_identification.v1` | `entity` | no | yes, recorded | never | open cross-modal **entity** identification |
| `mmpilot.open_animal_legs.v1` | `animal` | no | yes, recorded | never | leg-count recomputation *(with controls)* |
| `mmpilot.hidden_animal_legs.v1` | `animal` | no | never | never | multi-hop reasoning *(with controls)* |

**The task domain is not decoration.** *"What animal is present in the
evidence?"* cannot screen `toilet` or `microwave`, and both are in the pilot's
six-concept set: scoring them against that question measures what the model says
when asked for an animal that is not there, which is a different experiment with
a different interpretation. So every source, target and externally scored
identity under an `animal` protocol must carry the predeclared domain, and an
unspecified domain is a refusal rather than an assumption. A mixed category set
belongs to `open_entity_identification`, whose question presumes nothing — and
which supports no legs or multi-hop claim in exchange.

"How many legs" is animal-specific for the same reason, and it needs more: a
**unique registered leg count** for both the source and the target. An
unregistered or ambiguous concept has no ground truth and is refused.

An open prompt is **not** hidden-intermediate reasoning merely because the
candidate list is absent — `open_animal_identification` still lets the source
appear in the evidence.

## What this notebook is not

It never loads Gemma, never touches Drive, never reads a completed run, and
never produces a scientific result. It runs against the synthetic world in
`jlens.mmpilot.coordinate_swap_mock`, whose cross-modal concept vector is
**stipulated, not measured**. Every effect it prints is synthetic plumbing.

**MOCK SUCCESS IS NOT SCIENTIFIC EVIDENCE.** It says the plumbing computes what
it claims and nothing at all about Gemma, about any modality, or about open
identification, identity replacement, downstream recomputation, or multi-hop
reasoning.

Behavioral outputs stay **text**. `text`, `image` and `spoken_audio` are
*evidence* modalities, and `spoken_audio` means spoken captions, not
environmental sound. Identity replacement and downstream recomputation are
**separate claims**.
"""
)

markdown(
    """
## 1. Colab bootstrap

Constants only in the first cell; nothing from this repository is imported
until it has been installed.
"""
)

code(
    """
# 1a. Bootstrap constants only. Nothing from this repository is imported yet.
REPO_URL = "https://github.com/MechInterpreter/jacobian-lens-gemma.git"
BRANCH = "experiment/spokencoco-jspace-pilot"
REPO_DIR = "/content/jacobian-lens-gemma"

print(f"repo   {REPO_URL}")
print(f"branch {BRANCH}")
print(f"target {REPO_DIR}")
"""
)

code(
    '''
# 1b. Clone or update the repository, then verify the checked-out branch.
import os
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_PATH = Path(os.environ.get("MMPILOT_REPO_DIR") or REPO_DIR)


def _git(*arguments, cwd=None):
    result = subprocess.run(
        ["git", *arguments], cwd=cwd, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed:\\n{result.stdout}\\n{result.stderr}"
        )
    return result.stdout.strip()


if IN_COLAB:
    if not (REPO_PATH / ".git").is_dir():
        _git("clone", "--branch", BRANCH, REPO_URL, str(REPO_PATH))
    else:
        _git("fetch", "origin", BRANCH, cwd=REPO_PATH)
        _git("checkout", BRANCH, cwd=REPO_PATH)
        _git("reset", "--hard", f"origin/{BRANCH}", cwd=REPO_PATH)

CHECKED_OUT_BRANCH = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=REPO_PATH)
COMMIT = _git("rev-parse", "HEAD", cwd=REPO_PATH)
if IN_COLAB and CHECKED_OUT_BRANCH != BRANCH:
    raise RuntimeError(
        f"checked out {CHECKED_OUT_BRANCH!r}, expected {BRANCH!r} - "
        "refusing to continue against the wrong code"
    )
print(f"branch {CHECKED_OUT_BRANCH}")
print(f"commit {COMMIT}")
'''
)

code(
    '''
# 1c. Install the repository and verify that `import jlens` resolves here.
if IN_COLAB:
    print("installing the repository (editable) ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(REPO_PATH)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pip install -e failed:\\n{result.stdout[-2000:]}\\n{result.stderr[-2000:]}"
        )

os.chdir(REPO_PATH)
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

try:
    import jlens
except ModuleNotFoundError as error:
    raise RuntimeError(
        f"`import jlens` is still not importable after installation: {error}"
    ) from error

if Path(jlens.__file__).resolve().parent.parent != REPO_PATH.resolve():
    raise RuntimeError(f"`import jlens` resolved to {jlens.__file__}, not this checkout")
print(f"jlens  {jlens.__file__}")
print(f"cwd    {os.getcwd()}")
'''
)

markdown(
    """
## 2. The switch, and why it refuses

One switch, `False` in the committed notebook. Setting it to `True` does not
start a real experiment — it raises with the reason. Two things block the real
open-prompt study, and neither is settled here.
"""
)

code(
    '''
# 2. The one switch. False in the committed notebook, and True is a refusal.
RUN_REAL_OPEN_PROMPT_STUDY = False

if RUN_REAL_OPEN_PROMPT_STUDY:
    raise NotImplementedError(
        "The real open-prompt study does not exist yet, and this notebook will "
        "not improvise one.\\n\\n"
        "Blocking reason 1: it is the behavioural readout of the coordinate-swap "
        "experiment, whose protocol applies the swap over a CONTIGUOUS band of "
        "intermediate layers. The completed research-grade calibration confirmed "
        "layers 35, 38 and 40 on its untouched confirmation set - three isolated "
        "layers, not a band - so there is currently no admissible band.\\n\\n"
        "Blocking reason 2: an open prompt has to be shown to be answerable at "
        "all. Whether Gemma identifies a SpokenCOCO concept without a candidate "
        "list is an unmeasured behavioural question, and a floor-level open "
        "accuracy would make every downstream number uninterpretable.\\n\\n"
        "Until both are settled this notebook is MOCK-only by construction."
    )

MODE = "mock"
print(f"mode {MODE} - no model, no Drive, no completed run is read or written")
'''
)

code(
    """
# 3. Imports. Everything here is CPU-only and deterministic.
import torch

from jlens.mmpilot.capability import (
    DEFAULT_QUESTION,
    PROMPT_PROTOCOL_VERSION,
    build_prompt,
    build_question,
    prediction_and_margin,
    score_candidate_sequences,
)
from jlens.mmpilot.coordinate_swap import (
    CoordinateSwapError,
    assert_open_prompt_protocol,
    coordinate_swap_band,
)
from jlens.mmpilot.coordinate_swap_mock import (
    MOCK_MODEL_REVISION,
    MOCK_PROCESSOR_REVISION,
    PRIMARY_BAND,
    SwapMockBackend,
    mock_bases,
    mock_concept_tokens,
)
from jlens.mmpilot.prompt_protocol import (
    ANIMAL_LEGS_QUESTION,
    CANDIDATE_LISTED_IDENTIFICATION,
    CONCEPT_DOMAINS,
    DOMAIN_ANIMAL,
    HIDDEN_ANIMAL_LEGS,
    OPEN_ANIMAL_IDENTIFICATION,
    OPEN_ANIMAL_IDENTIFICATION_QUESTION,
    OPEN_ANIMAL_LEGS,
    OPEN_ENTITY_IDENTIFICATION,
    OPEN_ENTITY_IDENTIFICATION_QUESTION,
    RETIRED_PROTOCOLS,
    Evidence,
    PromptLeakageError,
    PromptProtocolError,
    PropertyAnswerError,
    PropertyContrastError,
    TaskDomainError,
    backend_input_kwargs,
    build_protocol_prompt,
    capability_filter_property_pairs,
    claim_admissibility_rule_record,
    concept_spec,
    normalize,
    prompt_protocol_fingerprint,
    protocol_claim_admissibility,
    resolve_leg_count,
    select_animal_concepts,
    select_property_contrast_pairs,
)

BACKEND = SwapMockBackend()
TOKENS = mock_concept_tokens(BACKEND)
SOURCE, TARGET = concept_spec("bird"), concept_spec("cat")
IDENTITY_CANDIDATES = ("bird", "cat")
PROPERTY_CANDIDATES = ("two", "four")
LEGACY_CONCEPTS = ["bird", "cat", "giraffe", "microwave", "toilet", "zebra"]

print(f"source {SOURCE.name!r} domain {SOURCE.domain!r} aliases {SOURCE.aliases}")
print(f"target {TARGET.name!r} domain {TARGET.domain!r} aliases {TARGET.aliases}")
print(f"legs   {SOURCE.name} {resolve_leg_count(SOURCE.name)}, "
      f"{TARGET.name} {resolve_leg_count(TARGET.name)}")
"""
)

markdown(
    """
## 4. The legacy candidate-listed mode still functions

Nothing about the completed protocol is renamed, rewritten, or reinterpreted.
The legacy question is rebuilt through the protocol module and compared
byte-for-byte against `jlens.mmpilot.capability`, which is what completed runs
actually called. Its leakage findings are **recorded, not hidden**: every
candidate is in that prompt and the record says so.
"""
)

code(
    '''
# 4. Legacy mode: same bytes, same protocol string, findings recorded.
LEGACY = build_protocol_prompt(
    protocol=CANDIDATE_LISTED_IDENTIFICATION,
    evidence=Evidence(modality="text", text="a bird on a branch"),
    external_candidates=LEGACY_CONCEPTS,
    legacy_candidate_list=LEGACY_CONCEPTS,
    encode_candidate=BACKEND.encode_candidate,
)
_expected = build_prompt(
    build_question(LEGACY_CONCEPTS), modality="text", caption="a bird on a branch"
)
LEGACY_BYTES_MATCH = LEGACY.model_visible_prompt == _expected
print(f"legacy template   {DEFAULT_QUESTION!r}")
print(f"legacy protocol   {PROMPT_PROTOCOL_VERSION}")
print(f"byte-for-byte     {LEGACY_BYTES_MATCH}")
print()
print(LEGACY.model_visible_prompt)
print()
print(f"candidates_in_prompt {LEGACY.candidate_visibility['candidates_in_prompt']}")
print(f"recorded findings    {LEGACY.leakage['recorded']}")
print(f"audit passed         {LEGACY.leakage['passed']}  (recorded is not refused)")
assert LEGACY_BYTES_MATCH
assert "instruction_candidate_leakage" in LEGACY.leakage["recorded"]
assert "candidate_enumeration_detected" in LEGACY.leakage["recorded"]
'''
)

code(
    '''
# 4b. And what that legacy prompt is allowed to support - only this.
LEGACY_CLAIM = protocol_claim_admissibility(
    protocol=CANDIDATE_LISTED_IDENTIFICATION, leakage=LEGACY.leakage, mode="real"
)
print(f"maximum claim  {LEGACY_CLAIM['maximum_claim']}")
for _excluded in LEGACY_CLAIM["excluded_claims"]:
    print(f"  never         {_excluded}")
assert LEGACY_CLAIM["maximum_claim"] == "candidate_conditioned_identification"
'''
)

markdown(
    """
## 5. The open prompt names no candidate

Three evidence channels, one question, byte-identical across all three. The
image and spoken-audio conditions carry *only* the question — the media is the
evidence.
"""
)

code(
    '''
# 5. One open identification prompt per evidence channel.
TRANSCRIPT = "a small bird perched on a branch"
EVIDENCE = {
    "text": Evidence(modality="text", text="A small bird perched on a branch."),
    "image": Evidence(
        modality="image", media="<pixels>", media_reference="/coco/bird_000001.jpg"
    ),
    "spoken_audio": Evidence(
        modality="spoken_audio",
        media=[0.0, 0.25, -0.25],
        transcript=TRANSCRIPT,
        sampling_rate=16000,
        media_reference="/spokencoco/wav/000001.wav",
    ),
}

OPEN = {
    name: build_protocol_prompt(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        evidence=evidence,
        external_candidates=IDENTITY_CANDIDATES,
        source=SOURCE,
        target=TARGET,
        encode_candidate=BACKEND.encode_candidate,
        encode_prompt=BACKEND.encode_token,
    )
    for name, evidence in EVIDENCE.items()
}

print("question (identical in every channel):")
print(OPEN_ANIMAL_IDENTIFICATION_QUESTION)
print()
for name, built in OPEN.items():
    _tokens = normalize(built.model_visible_prompt).split()
    _named = sorted(c for c in IDENTITY_CANDIDATES if c in _tokens)
    print(f"{name:<13} visible prompt = question: "
          f"{built.model_visible_prompt == OPEN_ANIMAL_IDENTIFICATION_QUESTION}")
    print(f"{'':<13} candidate names in the visible prompt: {_named or 'none'}")
assert all("cat" not in normalize(b.model_visible_prompt).split() for b in OPEN.values())
assert OPEN["image"].model_visible_prompt == OPEN_ANIMAL_IDENTIFICATION_QUESTION
assert OPEN["spoken_audio"].model_visible_prompt == OPEN_ANIMAL_IDENTIFICATION_QUESTION
'''
)

code(
    '''
# 5b. What each channel's audit found. Source recorded, target absent.
for name, built in OPEN.items():
    _findings = built.leakage["findings"]
    print(f"{name}:")
    for _category in (
        "source_in_visible_evidence",
        "target_in_visible_evidence",
        "source_in_audio_transcript",
        "target_in_audio_transcript",
        "semantic_filename_exposure",
        "instruction_candidate_leakage",
    ):
        _row = _findings[_category]
        _surfaces = ", ".join(hit["surface"] for hit in _row["matches"]) or "-"
        print(f"    {_category:<32} {_row['status']:<15} {_surfaces}")
    print()

# The written caption legitimately says "bird". That is recorded, not hidden.
assert OPEN["text"].leakage["source_in_visible_evidence"] is True
assert OPEN["text"].leakage["findings"]["source_in_visible_evidence"]["status"] == "recorded"
# The image condition has no text evidence at all - neither name is present.
assert OPEN["image"].leakage["source_in_visible_evidence"] is False
assert OPEN["image"].leakage["findings"]["target_in_visible_evidence"]["status"] == "clean"
# The spoken transcript was read by the audit and names the source, not the target.
assert OPEN["spoken_audio"].leakage["transcript_audited"] is True
assert OPEN["spoken_audio"].leakage["findings"]["source_in_audio_transcript"]["status"] == "recorded"
assert OPEN["spoken_audio"].leakage["findings"]["target_in_audio_transcript"]["status"] == "clean"
'''
)

markdown(
    """
## 6. Every refusal, exercised

Each of these would otherwise produce a plausible-looking number meaning
something other than what it would be reported as. The protocol is **refused**,
never quietly downgraded to a weaker one.
"""
)

code(
    '''
# 6. The refusals.
REFUSALS = {}


def _refused(label, exception_type, thunk):
    """Record *why* it was refused, not merely that it was."""
    try:
        thunk()
    except exception_type as error:
        _lines = [line.strip("- ").strip() for line in str(error).splitlines()]
        _reason = next((line for line in _lines[1:] if line), _lines[0])
        REFUSALS[label] = f"{type(error).__name__}: {_reason}"
        return
    raise AssertionError(f"{label} was NOT refused")


def _open(**overrides):
    kwargs = {
        "protocol": OPEN_ANIMAL_IDENTIFICATION,
        "evidence": Evidence(modality="text", text="A small bird on a branch."),
        "external_candidates": IDENTITY_CANDIDATES,
        "source": SOURCE,
        "target": TARGET,
    }
    kwargs.update(overrides)
    return build_protocol_prompt(**kwargs)


def _hidden(**overrides):
    kwargs = {
        "protocol": HIDDEN_ANIMAL_LEGS,
        "evidence": Evidence(
            modality="text", text="The animal in the evidence is the one that spins webs."
        ),
        "external_candidates": ("six", "eight"),
        "source": concept_spec("spider"),
        "target": concept_spec("ant"),
    }
    kwargs.update(overrides)
    return build_protocol_prompt(**kwargs)


_refused(
    "target_in_visible_evidence",
    PromptLeakageError,
    lambda: _open(evidence=Evidence(modality="text", text="A bird beside a cat.")),
)
_refused(
    "target_in_audio_transcript",
    PromptLeakageError,
    lambda: _open(
        evidence=Evidence(
            modality="spoken_audio",
            media=[0.0],
            transcript="a bird beside a cat",
            sampling_rate=16000,
        )
    ),
)
_refused(
    "unauditable_missing_transcript",
    PromptLeakageError,
    lambda: _open(
        evidence=Evidence(modality="spoken_audio", media=[0.0], sampling_rate=16000)
    ),
)
_refused(
    "hidden_intermediate_source_in_transcript",
    PromptLeakageError,
    lambda: _hidden(
        evidence=Evidence(
            modality="spoken_audio",
            media=[0.0],
            transcript="the animal that spins webs is a spider",
            sampling_rate=16000,
        )
    ),
)
_refused(
    "hidden_intermediate_target_in_transcript",
    PromptLeakageError,
    lambda: _hidden(
        evidence=Evidence(
            modality="spoken_audio",
            media=[0.0],
            transcript="it is certainly not an ant",
            sampling_rate=16000,
        )
    ),
)
_refused(
    "hidden_intermediate_source_in_visible_text",
    PromptLeakageError,
    lambda: _hidden(evidence=Evidence(modality="text", text="A spider on its web.")),
)
_refused(
    "property_answer_in_prompt",
    PromptLeakageError,
    lambda: build_protocol_prompt(
        protocol=OPEN_ANIMAL_LEGS,
        evidence=Evidence(modality="text", text="An animal with four legs."),
        external_candidates=PROPERTY_CANDIDATES,
        source=SOURCE,
        target=TARGET,
    ),
)
_refused(
    "semantic_filename_exposure",
    PromptLeakageError,
    lambda: _open(
        evidence=Evidence(
            modality="text",
            text="See bird_000001.jpg.",
            media_reference="/coco/bird_000001.jpg",
        )
    ),
)
_refused(
    "candidate_enumeration_in_the_open_question",
    PromptLeakageError,
    lambda: _open(
        evidence=Evidence(modality="image", media="<pixels>"),
        question="Which is it: aardvark, wombat, quokka?",
    ),
)

for label, message in sorted(REFUSALS.items()):
    print(f"{label:<42} {message[:96]}")
assert len(REFUSALS) == 9
'''
)

markdown(
    """
### The same evidence is legal under one protocol and refused under another

`open_identification` permits the source in natural written evidence and records
it. `hidden_intermediate` does not, because there is then no intermediate left
to be hidden. That is the distinction, and it is enforced rather than described.
"""
)

code(
    '''
# 6b. One caption, two protocols, two outcomes.
_caption = Evidence(modality="text", text="A small bird on a branch.")
_permitted = build_protocol_prompt(
    protocol=OPEN_ANIMAL_IDENTIFICATION,
    evidence=_caption,
    external_candidates=IDENTITY_CANDIDATES,
    source=SOURCE,
    target=TARGET,
)
print(f"open_identification : passed={_permitted.leakage['passed']}, "
      f"recorded={_permitted.leakage['recorded']}")
try:
    build_protocol_prompt(
        protocol=HIDDEN_ANIMAL_LEGS,
        evidence=_caption,
        external_candidates=PROPERTY_CANDIDATES,
        source=SOURCE,
        target=TARGET,
    )
except PromptLeakageError as error:
    PROTOCOL_SEPARATION = str(error).splitlines()[1].strip()
print(f"hidden_intermediate : {PROTOCOL_SEPARATION}")
assert "source_in_visible_evidence" in PROTOCOL_SEPARATION
'''
)

markdown(
    """
## 6c. The task domain, which the question already committed to

`What animal is present in the evidence?` presumes the answer is an animal. The
pilot's six-concept set contains `toilet` and `microwave`, so scoring that set
against that question would not be an open-identification test at all — it would
be a test of what the model says when asked for an animal that is not there.

The domain is therefore **declared, resolved and refused** rather than assumed,
and a mixed set goes to the domain-neutral protocol instead.
"""
)

code(
    '''
# 6c. The animal question refuses the mixed six-concept set, by name.
MIXED_SIX = ("bird", "cat", "giraffe", "microwave", "toilet", "zebra")
try:
    build_protocol_prompt(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        evidence=Evidence(modality="image", media="<pixels>"),
        external_candidates=MIXED_SIX,
        source=SOURCE,
        target=TARGET,
    )
except TaskDomainError as error:
    MIXED_SET_REFUSAL = str(error).splitlines()[0]

print("declared domains:")
for _name in MIXED_SIX:
    print(f"  {_name:<10} {CONCEPT_DOMAINS.get(_name) or '(unspecified)'}")
print()
print(f"open_animal_identification -> {MIXED_SET_REFUSAL[:160]}")
assert "toilet is domain 'furniture'" in MIXED_SET_REFUSAL
assert "microwave is domain 'appliance'" in MIXED_SET_REFUSAL

# An unspecified domain is refused too - it is not read as "probably fine".
try:
    build_protocol_prompt(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        evidence=Evidence(modality="image", media="<pixels>"),
        external_candidates=("bird", "cat", "wombat"),
        source=SOURCE,
        target=TARGET,
    )
except TaskDomainError as error:
    UNSPECIFIED_DOMAIN_REFUSAL = str(error).splitlines()[0]
print(f"unregistered concept       -> "
      f"{UNSPECIFIED_DOMAIN_REFUSAL.split('must be in it.')[-1].strip()[:110]}")
assert "wombat has no registered domain" in UNSPECIFIED_DOMAIN_REFUSAL
'''
)

code(
    '''
# 6d. The same mixed set under the domain-neutral protocol: accepted, recorded.
ENTITY = build_protocol_prompt(
    protocol=OPEN_ENTITY_IDENTIFICATION,
    evidence=Evidence(modality="image", media="<pixels>"),
    external_candidates=MIXED_SIX,
    source=SOURCE,
    target=TARGET,
    encode_candidate=BACKEND.encode_candidate,
)
print(f"question: {ENTITY.model_visible_prompt.splitlines()[0]}")
print(f"passed:   {ENTITY.leakage['passed']}")
print(f"domains observed and recorded: {ENTITY.task_domain['observed_domains']}")
print()
ENTITY_CLAIM = protocol_claim_admissibility(
    protocol=OPEN_ENTITY_IDENTIFICATION, leakage=ENTITY.leakage, mode="real"
)
print(f"maximum claim: {ENTITY_CLAIM['maximum_claim']}")
for _excluded in ENTITY_CLAIM["excluded_claims"]:
    print(f"  never        {_excluded}")
assert "animal" not in normalize(ENTITY.model_visible_prompt).split()
assert ENTITY.task_domain["observed_domains"] == ["animal", "appliance", "furniture"]
assert "multi-hop reasoning" in ENTITY_CLAIM["excluded_claims"]
'''
)

code(
    '''
# 6e. The legs protocols need an animal AND a unique registered leg count.
DOMAIN_REFUSALS = {}


def _legs(**overrides):
    kwargs = {
        "protocol": OPEN_ANIMAL_LEGS,
        "evidence": Evidence(modality="image", media="<pixels>"),
        "external_candidates": PROPERTY_CANDIDATES,
        "source": SOURCE,
        "target": TARGET,
    }
    kwargs.update(overrides)
    return build_protocol_prompt(**kwargs)


for _label, _kwargs, _expected in (
    ("non-animal target", {"target": concept_spec("toilet")}, TaskDomainError),
    (
        "animal with no registered leg count",
        {"target": concept_spec("dolphin", domain=DOMAIN_ANIMAL)},
        PropertyAnswerError,
    ),
    (
        "ambiguous leg count",
        {"leg_counts": {"bird": (2,), "cat": (2, 4)}},
        PropertyAnswerError,
    ),
    ("unregistered answer choice", {"external_candidates": ("two", "seven")}, PropertyAnswerError),
):
    try:
        _legs(**_kwargs)
        raise AssertionError(f"{_label} was NOT refused")
    except _expected as error:
        DOMAIN_REFUSALS[_label] = f"{type(error).__name__}: {str(error).splitlines()[0]}"

# And the retired, domain-blind names name their replacements.
for _old, _new in RETIRED_PROTOCOLS.items():
    try:
        _legs(protocol=_old)
        raise AssertionError(f"{_old} was NOT refused")
    except PromptProtocolError as error:
        DOMAIN_REFUSALS[f"retired {_old}"] = str(error).splitlines()[0]

for _label, _message in sorted(DOMAIN_REFUSALS.items()):
    print(f"{_label:<46} {_message[:100]}")
assert len(DOMAIN_REFUSALS) == 7
'''
)

markdown(
    """
## 6f. The predeclared animal concept set

The first real coordinate-swap study needs animal-only concepts, and they have
to be chosen **before** any model result. `select_animal_concepts` filters the
rows the existing deterministic ranking and evidence audit already produce — it
re-implements neither — by domain, by feasibility, and by whether a unique leg
count is registered, preserving ranking order throughout.

`bird`, `cat`, `giraffe`, `zebra`, `sheep` and `cow` are the *likely* survivors
from SpokenCOCO. They are **not assumed**: coverage is whatever the local
annotation files support, and short coverage is a refusal rather than a gap to
fill. The rows below are synthetic, standing in for the real ranking.
"""
)

code(
    '''
# 6f. Domain, feasibility, property - applied to a stand-in ranking.
MOCK_RANKING = [
    {"concept": "bird", "feasible": True, "unmet": []},
    {"concept": "toilet", "feasible": True, "unmet": []},
    {"concept": "cat", "feasible": True, "unmet": []},
    {"concept": "microwave", "feasible": True, "unmet": []},
    {"concept": "giraffe", "feasible": False, "unmet": ["distinct_images 3 < 8"]},
    {"concept": "zebra", "feasible": True, "unmet": []},
    {"concept": "sheep", "feasible": True, "unmet": []},
    {"concept": "cow", "feasible": True, "unmet": []},
]
ANIMAL_SELECTION = select_animal_concepts(MOCK_RANKING, n_focal=2)

print(f"ranked input      {ANIMAL_SELECTION['ranked_input']}")
print(f"animal concepts   {ANIMAL_SELECTION['animal_concepts']}   (ranking order kept)")
print(f"focal             {ANIMAL_SELECTION['focal']}")
print(f"unrelated control {ANIMAL_SELECTION['non_focal']}")
print(f"leg counts        {ANIMAL_SELECTION['leg_counts']}")
print()
for _row in ANIMAL_SELECTION["excluded"]:
    print(f"  dropped {_row['concept']:<10} at {_row['stage']:<14} {_row['reason'][:60]}")
print()
print(f"selection checksum {ANIMAL_SELECTION['selection_checksum']}")

# Coverage is never assumed: too few animals is a refusal, not a smaller study.
try:
    select_animal_concepts(
        [{"concept": "bird", "feasible": True}, {"concept": "toilet", "feasible": True}]
    )
except PromptProtocolError as error:
    COVERAGE_REFUSAL = str(error).splitlines()[0]
print(f"short coverage -> {COVERAGE_REFUSAL[:120]}")

# And it cannot be chosen from rows that already know how the model behaved.
try:
    select_animal_concepts([{"concept": "bird", "feasible": True, "accuracy": 0.875}])
except PromptProtocolError as error:
    POST_MODEL_REFUSAL = str(error).splitlines()[0]
print(f"post-model rows -> {POST_MODEL_REFUSAL[:120]}")
assert ANIMAL_SELECTION["animal_concepts"] == ["bird", "cat", "zebra", "sheep", "cow"]
assert "Coverage is" in COVERAGE_REFUSAL
assert "post-model field" in POST_MODEL_REFUSAL
'''
)

markdown(
    """
### Property pairs are selected separately

Identity replacement may compare any two animals. Leg-count recomputation may
not: its source and target must imply different answers. The directed pair and
its unrelated control are fixed from the pre-model ranking, then capability may
exclude the pair but can never replace it.
"""
)

code(
    '''
# 6g. The first unequal-property pair in ranking order, emitted both ways.
PROPERTY_PAIRS = select_property_contrast_pairs(ANIMAL_SELECTION)
print(f"pair rule {PROPERTY_PAIRS['pair_selection_version']}")
for _pair in PROPERTY_PAIRS["ordered_directed_pairs"]:
    print(
        f"  {_pair['source']}({_pair['source_property_value']}) -> "
        f"{_pair['target']}({_pair['target_property_value']}); "
        f"control={_pair['unrelated_control']} "
        f"({_pair['unrelated_control_relation']})"
    )
assert [
    (_row["source"], _row["target"])
    for _row in PROPERTY_PAIRS["ordered_directed_pairs"]
] == [("bird", "cat"), ("cat", "bird")]

# Capability failure excludes the fixed pair and never substitutes zebra.
CAPABILITY_FILTER = capability_filter_property_pairs(
    PROPERTY_PAIRS, eligible_concepts=("cat", "zebra", "sheep", "cow")
)
assert CAPABILITY_FILTER["eligible_pairs"] == []
assert [
    (_row["source"], _row["target"])
    for _row in CAPABILITY_FILTER["pairs"]
] == [("bird", "cat"), ("cat", "bird")]

# A pool with only four-legged animals cannot support this property study.
try:
    select_property_contrast_pairs(
        select_animal_concepts(
            [
                {"concept": "cat", "feasible": True},
                {"concept": "zebra", "feasible": True},
                {"concept": "sheep", "feasible": True},
            ],
            n_focal=2,
        )
    )
except PropertyContrastError as error:
    NO_CONTRAST_REFUSAL = str(error).splitlines()[0]
print(f"all-four-leg pool -> {NO_CONTRAST_REFUSAL[:120]}")
'''
)

markdown(
    """
## 7. The transcript is audited and never reaches the model

The spoken-audio condition's whole claim is that the recording is the only
evidence. The transcript is read by the offline audit and by nothing else — so
the backend call is intercepted here and its arguments are inspected.
"""
)

code(
    '''
# 7. Intercept the backend call and prove what crossed the boundary.
SEEN_BACKEND_CALLS = []


class _RecordingBackend:
    """Wraps the mock backend and records exactly what it was handed."""

    def __init__(self, inner):
        self._inner = inner

    def build_inputs(self, **kwargs):
        SEEN_BACKEND_CALLS.append(dict(kwargs))
        return self._inner.build_inputs(**kwargs)


_audio = OPEN["spoken_audio"]
_kwargs = backend_input_kwargs(_audio, transcript=TRANSCRIPT)
_RecordingBackend(BACKEND).build_inputs(**_kwargs)

TRANSCRIPT_CROSSED = any(
    TRANSCRIPT in str(value) for call in SEEN_BACKEND_CALLS for value in call.values()
)
print(f"kwargs handed to the backend: {sorted(_kwargs)}")
print(f"prompt passed               : {_kwargs['prompt']!r}")
print(f"transcript in any argument  : {TRANSCRIPT_CROSSED}")
print(f"transcript recorded as hash : {_audio.transcript_hash}")
print(f"transcript in the artifact  : {TRANSCRIPT in str(_audio.to_dict())}")
assert TRANSCRIPT_CROSSED is False
assert TRANSCRIPT not in str(_audio.to_dict())
'''
)

markdown(
    """
## 8. The candidates are scored externally, and order does not matter

The scorer appends each candidate's **complete** token sequence after the prompt
and reads the teacher-forced conditional log likelihood. The prompt is untouched
by that, so reversing the candidate order changes neither the prompt tokens nor
any candidate's own score.
"""
)

code(
    '''
# 8. External complete-sequence scoring, forward and reversed.
IDENTITY_INPUTS = BACKEND.build_inputs(
    prompt=OPEN["image"].model_visible_prompt, modality="image", concept="bird"
)
FORWARD_IDS = dict(OPEN["image"].external_candidate_token_ids)
REVERSE_IDS = dict(reversed(list(FORWARD_IDS.items())))

FORWARD_SCORES = score_candidate_sequences(BACKEND, IDENTITY_INPUTS, FORWARD_IDS)
REVERSE_SCORES = score_candidate_sequences(BACKEND, IDENTITY_INPUTS, REVERSE_IDS)

for name, ids in FORWARD_IDS.items():
    _delta = abs(FORWARD_SCORES[name]["sum_logprob"] - REVERSE_SCORES[name]["sum_logprob"])
    print(f"{name:<6} tokens {ids}  n={FORWARD_SCORES[name]['n_tokens']}  "
          f"sum_logprob {FORWARD_SCORES[name]['sum_logprob']:+.6f}  "
          f"|delta| under reversal {_delta:.2e}")

ORDER_INVARIANCE_MAX_DELTA = max(
    abs(FORWARD_SCORES[n]["sum_logprob"] - REVERSE_SCORES[n]["sum_logprob"])
    for n in FORWARD_IDS
)
assert all(len(ids) > 1 for ids in FORWARD_IDS.values()), "multi-token, on purpose"
assert ORDER_INVARIANCE_MAX_DELTA < 1e-9
'''
)

code(
    '''
# 8b. Reversing the candidate order also leaves the prompt itself alone.
def _built(candidates):
    return build_protocol_prompt(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        evidence=Evidence(modality="image", media="<pixels>"),
        external_candidates=candidates,
        source=SOURCE,
        target=TARGET,
        encode_candidate=BACKEND.encode_candidate,
        encode_prompt=BACKEND.encode_token,
    )


_forward = _built(("bird", "cat"))
_reversed = _built(("cat", "bird"))
PROMPT_ORDER_INVARIANT = (
    _forward.prompt_hash == _reversed.prompt_hash
    and _forward.prompt_token_ids == _reversed.prompt_token_ids
)
print(f"prompt hash forward  {_forward.prompt_hash}")
print(f"prompt hash reversed {_reversed.prompt_hash}")
print(f"prompt token ids identical: {PROMPT_ORDER_INVARIANT}")
assert PROMPT_ORDER_INVARIANT
'''
)

markdown(
    """
## 9. Order must not move the fingerprint; the candidate **set** must

Two different requirements pulling in opposite directions, and both are checked.
A reordered candidate list is the same measurement. A different candidate list
is not.
"""
)

code(
    '''
# 9. Fingerprint sensitivity.
def _fingerprint(candidates):
    return prompt_protocol_fingerprint(
        _built(candidates),
        model_revision=MOCK_MODEL_REVISION,
        processor_revision=MOCK_PROCESSOR_REVISION,
    )


_two = _fingerprint(("bird", "cat"))
_two_reversed = _fingerprint(("cat", "bird"))
_three = _fingerprint(("bird", "cat", "giraffe"))

print(f"{'candidates':<26} {'prompt hash':<20} prompt-protocol digest")
for _label, _payload in (
    ("bird, cat", _two),
    ("cat, bird (reordered)", _two_reversed),
    ("bird, cat, giraffe", _three),
):
    print(f"{_label:<26} {_payload['prompt_hash']:<20} {_payload['prompt_protocol_digest']}")

def _differs(left, right, key):
    return left[key] != right[key]


FINGERPRINT_SENSITIVITY = {
    "order_changes_digest": _differs(_two, _two_reversed, "prompt_protocol_digest"),
    "set_changes_digest": _differs(_two, _three, "prompt_protocol_digest"),
    "set_changes_prompt_hash": _differs(_two, _three, "prompt_hash"),
}
print()
print(FINGERPRINT_SENSITIVITY)
assert FINGERPRINT_SENSITIVITY == {
    "order_changes_digest": False,
    "set_changes_digest": True,
    "set_changes_prompt_hash": False,
}
'''
)

code(
    '''
# 9b. And the primary coordinate-swap study refuses a candidate-listed prompt.
LEGACY_FINGERPRINT = prompt_protocol_fingerprint(
    LEGACY, model_revision=MOCK_MODEL_REVISION, processor_revision=MOCK_PROCESSOR_REVISION
)
try:
    assert_open_prompt_protocol(LEGACY_FINGERPRINT)
except CoordinateSwapError as error:
    PRIMARY_PROTOCOL_REFUSAL = str(error).splitlines()[0]
print(f"candidate-listed -> {PRIMARY_PROTOCOL_REFUSAL[:150]}")
print(f"open             -> admissible: "
      f"{assert_open_prompt_protocol(_two)['prompt_protocol_version']}")
assert "not admissible for the primary" in PRIMARY_PROTOCOL_REFUSAL
'''
)

markdown(
    """
## 10. Candidate-completion positions are never patched

The swap runs at every *prompt* position. The teacher-forced candidate tokens
sit past `prompt_len`, and no position rule can reach them — checked here
against the hook's own record and against the activations themselves.
"""
)

code(
    '''
# 10. The prompt/candidate boundary, under an active swap.
BASES = mock_bases(
    BACKEND.world, layers=PRIMARY_BAND, source=TOKENS["bird"], target=TOKENS["cat"]
)
with coordinate_swap_band(
    BACKEND.blocks,
    BASES,
    alpha=1.0,
    prompt_len=IDENTITY_INPUTS.prompt_len,
    position_rule="all_prompt_positions",
    record_coordinates=False,
) as _stats:
    SWAPPED_SCORES = score_candidate_sequences(BACKEND, IDENTITY_INPUTS, FORWARD_IDS)

for _layer, _row in sorted(_stats.items()):
    print(f"L{_layer}: patched positions 0..{max(_row['positions'])}  "
          f"prompt_len {_row['prompt_len']}  "
          f"candidate positions skipped {_row['n_candidate_positions_skipped']}")
assert all(
    max(row["positions"]) == IDENTITY_INPUTS.prompt_len - 1 for row in _stats.values()
)
assert all(row["n_candidate_positions_skipped"] > 0 for row in _stats.values())
'''
)

code(
    '''
# 10b. The same thing measured on the activations, not on the bookkeeping.
_ids = FORWARD_IDS["cat"]
_tensors = dict(IDENTITY_INPUTS.tensors)
_tensors["input_ids"] = torch.cat(
    [_tensors["input_ids"], torch.tensor([list(_ids)], dtype=torch.long)], dim=1
)
_tensors["attention_mask"] = torch.ones(1, _tensors["input_ids"].shape[1], dtype=torch.long)
_captured = {}


def _record(_module, _inputs, output):
    _captured["hidden"] = (output if torch.is_tensor(output) else output[0]).clone()


_handle = BACKEND.blocks[max(PRIMARY_BAND)].register_forward_hook(_record)
try:
    BACKEND.forward_logits(_tensors)
    _clean_tail = _captured["hidden"][0, IDENTITY_INPUTS.prompt_len:].clone()
    with coordinate_swap_band(
        BACKEND.blocks,
        BASES,
        alpha=1.0,
        prompt_len=IDENTITY_INPUTS.prompt_len,
        position_rule="all_prompt_positions",
        record_coordinates=False,
    ):
        BACKEND.forward_logits(_tensors)
    _patched_tail = _captured["hidden"][0, IDENTITY_INPUTS.prompt_len:]
finally:
    _handle.remove()

CANDIDATE_TAIL_BIT_IDENTICAL = bool(torch.equal(_clean_tail, _patched_tail))
print(f"candidate-completion activations bit-identical under the swap: "
      f"{CANDIDATE_TAIL_BIT_IDENTICAL}")
assert CANDIDATE_TAIL_BIT_IDENTICAL
'''
)

markdown(
    """
## 11. The MOCK effects — synthetic plumbing, labelled as such

The identity moves and the property follows it, because this world was **built**
so that it would: the concept vector is shared across channels by construction
and the legs answer is computed from the identity coordinates by a hand-written
layer. These numbers demonstrate that the open prompt, the external scorer and
the swap compose. They are evidence about wiring and about nothing else.
"""
)

code(
    '''
# 11. Identity and property under the open prompts. Synthetic, by construction.
PROPERTY_PROMPT = build_protocol_prompt(
    protocol=OPEN_ANIMAL_LEGS,
    evidence=Evidence(modality="image", media="<pixels>"),
    external_candidates=PROPERTY_CANDIDATES,
    source=SOURCE,
    target=TARGET,
    encode_candidate=BACKEND.encode_candidate,
)
PROPERTY_INPUTS = BACKEND.build_inputs(
    prompt=PROPERTY_PROMPT.model_visible_prompt, modality="image", concept="bird"
)
PROPERTY_IDS = dict(PROPERTY_PROMPT.external_candidate_token_ids)

_clean_property = score_candidate_sequences(BACKEND, PROPERTY_INPUTS, PROPERTY_IDS)
with coordinate_swap_band(
    BACKEND.blocks,
    BASES,
    alpha=1.0,
    prompt_len=PROPERTY_INPUTS.prompt_len,
    position_rule="all_prompt_positions",
    record_coordinates=False,
):
    _swapped_property = score_candidate_sequences(BACKEND, PROPERTY_INPUTS, PROPERTY_IDS)

MOCK_EFFECTS = {
    "identity_clean": prediction_and_margin(FORWARD_SCORES, "cat")["prediction"],
    "identity_swapped": prediction_and_margin(SWAPPED_SCORES, "cat")["prediction"],
    "property_clean": prediction_and_margin(_clean_property, "four")["prediction"],
    "property_swapped": prediction_and_margin(_swapped_property, "four")["prediction"],
}
print("open identification question:")
print(f"  {OPEN_ANIMAL_IDENTIFICATION_QUESTION.splitlines()[0]}")
print(f"  clean -> {MOCK_EFFECTS['identity_clean']},  "
      f"bird->cat swap -> {MOCK_EFFECTS['identity_swapped']}")
print("open downstream-property question:")
print(f"  {ANIMAL_LEGS_QUESTION.splitlines()[0]}")
print(f"  clean -> {MOCK_EFFECTS['property_clean']},  "
      f"same swap -> {MOCK_EFFECTS['property_swapped']}")
print()
print("SYNTHETIC PLUMBING ONLY. The shared cross-modal concept vector is")
print("stipulated and the property layer is hand-written, so these outcomes are")
print("closed-form consequences of the construction - not findings.")
assert MOCK_EFFECTS == {
    "identity_clean": "bird",
    "identity_swapped": "cat",
    "property_clean": "two",
    "property_swapped": "four",
}
'''
)

code(
    '''
# 11b. And the claim rule says exactly that, whatever the numbers were.
MOCK_CLAIMS = {
    protocol: protocol_claim_admissibility(
        protocol=protocol,
        leakage=leakage,
        mode=MODE,
        property_contrast=(
            PROPERTY_PROMPT.property_contrast
            if protocol == OPEN_ANIMAL_LEGS
            else None
        ),
        identity_replacement_passed=True,
        direct_answer_control_passed=True,
        direct_answer_onset_control_passed=True,
    )
    for protocol, leakage in (
        (OPEN_ANIMAL_IDENTIFICATION, OPEN["image"].leakage),
        (OPEN_ANIMAL_LEGS, PROPERTY_PROMPT.leakage),
    )
}
for protocol, decision in MOCK_CLAIMS.items():
    print(f"{protocol}")
    print(f"    admissible    {decision['admissible']}")
    print(f"    granted claim {decision['granted_claim']}")
    print(f"    because       {decision['reasons'][0]}")
assert all(d["admissible"] is False for d in MOCK_CLAIMS.values())
assert all(d["granted_claim"] is None for d in MOCK_CLAIMS.values())
'''
)

markdown(
    """
## 12. Summary
"""
)

code(
    '''
# 12. Collect everything this MOCK run established.
CLAIM_RULE = claim_admissibility_rule_record()
SUMMARY = {
    "mode": MODE,
    "ran_real_experiment": False,
    "loaded_gemma": "transformers" in sys.modules,
    "legacy": {
        "protocol": CANDIDATE_LISTED_IDENTIFICATION,
        "capability_protocol_string": PROMPT_PROTOCOL_VERSION,
        "byte_for_byte_with_capability_module": LEGACY_BYTES_MATCH,
        "candidates_in_prompt": LEGACY.candidate_visibility["candidates_in_prompt"],
        "maximum_claim": LEGACY_CLAIM["maximum_claim"],
    },
    "open_identification": {
        "question": OPEN_ANIMAL_IDENTIFICATION_QUESTION,
        "visible_prompt_by_modality": {
            name: built.model_visible_prompt for name, built in OPEN.items()
        },
        "candidates_are_external": {
            name: built.candidates_are_external for name, built in OPEN.items()
        },
        "source_in_visible_evidence": {
            name: built.leakage["source_in_visible_evidence"]
            for name, built in OPEN.items()
        },
        "target_detected_anywhere": {
            name: built.leakage["findings"]["target_in_visible_evidence"]["detected"]
            or built.leakage["findings"]["target_in_audio_transcript"]["detected"]
            for name, built in OPEN.items()
        },
    },
    "open_downstream_property_question": ANIMAL_LEGS_QUESTION,
    "refusals": dict(sorted(REFUSALS.items())),
    "protocol_separation": PROTOCOL_SEPARATION,
    "task_domain": {
        "mixed_set_refusal": MIXED_SET_REFUSAL,
        "unspecified_domain_refusal": UNSPECIFIED_DOMAIN_REFUSAL,
        "domain_refusals": dict(sorted(DOMAIN_REFUSALS.items())),
        "entity_question": ENTITY.model_visible_prompt,
        "entity_observed_domains": ENTITY.task_domain["observed_domains"],
        "entity_maximum_claim": ENTITY_CLAIM["maximum_claim"],
        "entity_excluded_claims": ENTITY_CLAIM["excluded_claims"],
        "animal_question_domain": OPEN["image"].task_domain["task_domain"],
        "legs_property_schema": PROPERTY_PROMPT.property_schema,
        "legs_answers": PROPERTY_PROMPT.property_answers,
    },
    "animal_concept_selection": {
        "selection_version": ANIMAL_SELECTION["selection_version"],
        "animal_concepts": ANIMAL_SELECTION["animal_concepts"],
        "focal": ANIMAL_SELECTION["focal"],
        "non_focal": ANIMAL_SELECTION["non_focal"],
        "leg_counts": ANIMAL_SELECTION["leg_counts"],
        "excluded": ANIMAL_SELECTION["excluded"],
        "selection_checksum": ANIMAL_SELECTION["selection_checksum"],
        "coverage_refusal": COVERAGE_REFUSAL,
        "post_model_refusal": POST_MODEL_REFUSAL,
    },
    "property_contrast_pair_selection": {
        "pair_selection_version": PROPERTY_PAIRS["pair_selection_version"],
        "ordered_directed_pairs": PROPERTY_PAIRS["ordered_directed_pairs"],
        "pair_selection_checksum": PROPERTY_PAIRS["pair_selection_checksum"],
        "capability_filter_version": CAPABILITY_FILTER["filter_version"],
        "capability_does_not_replace": CAPABILITY_FILTER["replacement_forbidden"],
        "all_four_leg_pool_refusal": NO_CONTRAST_REFUSAL,
    },
    "transcript_reached_backend": TRANSCRIPT_CROSSED,
    "external_scoring": {
        "candidate_token_ids": {k: list(v) for k, v in FORWARD_IDS.items()},
        "all_multi_token": all(len(v) > 1 for v in FORWARD_IDS.values()),
        "order_invariance_max_abs_delta": ORDER_INVARIANCE_MAX_DELTA,
        "prompt_order_invariant": PROMPT_ORDER_INVARIANT,
    },
    "fingerprint_sensitivity": FINGERPRINT_SENSITIVITY,
    "primary_protocol_refusal": PRIMARY_PROTOCOL_REFUSAL,
    "candidate_tail_bit_identical": CANDIDATE_TAIL_BIT_IDENTICAL,
    "mock_effects": MOCK_EFFECTS,
    "mock_effects_are": "synthetic plumbing only - not scientific evidence",
    "mock_claims_admissible": {p: d["admissible"] for p, d in MOCK_CLAIMS.items()},
    "claim_rule_checksum": CLAIM_RULE["rule_checksum"],
}
STATUS = "MOCK_PASSED"

print("=" * 72)
print(f"{STATUS} - the open-prompt protocol computes what it says it computes")
print("=" * 72)
print(f"legacy prompt reproduced byte-for-byte   {LEGACY_BYTES_MATCH}")
print(f"open prompts name no candidate           True")
print(f"transcript reached the backend           {TRANSCRIPT_CROSSED}")
print(f"candidate order moved a score            "
      f"{ORDER_INVARIANCE_MAX_DELTA > 1e-9}")
print(f"candidate set moved the fingerprint      "
      f"{FINGERPRINT_SENSITIVITY['set_changes_digest']}")
print(f"candidate positions patched              "
      f"{not CANDIDATE_TAIL_BIT_IDENTICAL}")
print(f"refusals exercised                       "
      f"{len(REFUSALS) + len(DOMAIN_REFUSALS)}")
print(f"animal question screened the mixed set   True")
print(f"predeclared animal concepts              "
      f"{ANIMAL_SELECTION['animal_concepts']}")
print()
print("MOCK SUCCESS IS NOT SCIENTIFIC EVIDENCE. This says the plumbing is")
print("correct and nothing at all about Gemma, about any modality, or about")
print("open identification, identity replacement, downstream recomputation or")
print("multi-hop reasoning. Those require the real study, which has not run.")
'''
)


def build() -> dict:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": text.splitlines(keepends=True),
                **({"execution_count": None, "outputs": []} if kind == "code" else {}),
            }
            for kind, text in CELLS
        ],
        "metadata": {
            "accelerator": "None",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    TARGET.write_text(
        json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {TARGET}")
