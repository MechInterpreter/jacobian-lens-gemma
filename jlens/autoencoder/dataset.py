# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Building the (phrase, cone) dataset from a public corpus.

One dataset record = one **occurrence** of one phrase:

* the phrase text and its Gemma token ids (segmented the way the verbalizer will
  actually emit them — as the assistant continuation of the constant prompt, not
  in isolation),
* where it came from (document hash, character span),
* the layer-14 activation captured **immediately before the phrase begins**,
* the ``k=10`` nonnegative pursuit against the layer's J-space dictionary: the
  active atoms, their coefficients, and the weighted reconstruction ``q``,
* hashes of the activation and of ``q``, so any later claim about which record
  produced which vector is checkable by equality.

Splitting is by **phrase identity**, never by occurrence: every occurrence of a
phrase lands in the same split, assigned by rank of a salted hash of the
normalized phrase (:func:`assign_splits`), so the assignment depends only on the
*set* of mined phrases — not on corpus order, mining order, or which occurrence
happened to be seen first — and no split can come out empty.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field

import torch

from jlens.autoencoder.config import SPLITS, DatasetConfig, PursuitConfig
from jlens.autoencoder.errors import AutoencoderError
from jlens.generative import tensor_sha256, weighted_reconstruction
from jlens.hooks import ActivationRecorder
from jlens.pursuit import JSpaceDictionary, PursuitSettings, gradient_pursuit

DATASET_RECORD_SCHEMA = "jlens.autoencoder.dataset.record.v1"
DATASET_MANIFEST_SCHEMA = "jlens.autoencoder.dataset.manifest.v1"

#: Words that cannot carry a phrase on their own. A phrase consisting only of
#: these, or starting/ending with one, is generic scaffolding rather than a
#: nameable thing, and would give the reconstructor an easy target with no
#: semantics behind it.
FUNCTION_WORDS = frozenset(
    """
    a an the this that these those and or but nor for so yet of in on at by to from
    with without within into onto over under above below between among during before
    after since until while as is are was were be been being am do does did done
    have has had having will would shall should can could may might must not no
    it its it's he she they them his her their our your my me we you i who whom
    whose which what when where why how there here then than also very more most
    much many some any each every both few several other another such same own
    one two three four five six seven eight nine ten first second third new old
    """.split()
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def normalize_phrase(text: str) -> str:
    """Casefolded, whitespace-collapsed form used for identity and comparison."""
    return " ".join(str(text).split()).casefold()


def phrase_id(text: str) -> str:
    """Stable short id for a phrase, independent of corpus order."""
    return hashlib.sha256(normalize_phrase(text).encode()).hexdigest()[:16]


def document_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(str(text).encode()).hexdigest()


def split_position(text: str, *, salt: str) -> float:
    """The phrase's position in ``[0, 1)`` — a salted hash of its normalized form.

    Deterministic and independent of corpus order, dataset size, and the order
    phrases happened to be mined in.
    """
    digest = hashlib.sha256(f"{salt}|{normalize_phrase(text)}".encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def assign_splits(
    phrases: Iterable[str],
    *,
    salt: str,
    val_fraction: float,
    heldout_fraction: float,
) -> dict[str, str]:
    """Assign every distinct phrase to a split by **hash rank**.

    Returns ``{normalized phrase: split}``.

    Plain hash bucketing (``position < heldout_fraction`` ⇒ held out) is the
    obvious rule and was the first implementation, but at feasibility-study sizes
    it does not work: the phrase set is not a random sample — it is the top-N by
    corpus frequency — and a 32-phrase smoke build was observed with *zero*
    held-out phrases, which silently removes the only split the experiment's
    conclusions are allowed to rest on. Ranking the same hash positions and
    cutting at the quantiles keeps the assignment deterministic and
    order-independent (it depends on the *set* of phrases, not their order) while
    guaranteeing every split is non-empty whenever there are at least three
    phrases.

    The cost is that the assignment depends on the whole phrase set, so adding
    phrases can move an existing one. That is acceptable because a dataset is
    built once and its splits are stored in the records; the recomputation in
    :func:`assert_no_split_leakage` uses the stored set, so the check stays exact.
    """
    normalized = sorted({normalize_phrase(p) for p in phrases if str(p).strip()})
    total = len(normalized)
    if total == 0:
        return {}
    ordered = sorted(normalized, key=lambda p: (split_position(p, salt=salt), p))
    if total < 3:
        # Too few to stratify; everything trains, and assert_usable() will refuse
        # to let such a dataset be evaluated on.
        return {phrase: "train" for phrase in ordered}
    n_heldout = max(1, int(round(total * float(heldout_fraction))))
    n_val = max(1, int(round(total * float(val_fraction))))
    while n_heldout + n_val >= total:
        if n_val > 1:
            n_val -= 1
        elif n_heldout > 1:
            n_heldout -= 1
        else:  # pragma: no cover - unreachable for total >= 3
            break
    assignment: dict[str, str] = {}
    for index, phrase in enumerate(ordered):
        if index < n_heldout:
            assignment[phrase] = "heldout"
        elif index < n_heldout + n_val:
            assignment[phrase] = "val"
        else:
            assignment[phrase] = "train"
    return assignment


def _candidate_spans(sentence: str) -> list[tuple[int, int, str]]:
    """Candidate 2-4 word phrases in one sentence, as ``(start, end, text)``.

    Two families, both cheap and dependency-free:

    * **Named entities** — runs of capitalized words not at sentence start.
    * **Noun phrases** — content-word bigrams/trigrams whose first and last
      words are not function words.

    This is a heuristic miner, not a parser. Precision is bought back by the
    downstream filters (token count, function words, occurrence count), and the
    cost of a bad candidate is a phrase nobody can name — which shows up as a
    low reconstructor score, not as a silent bias.
    """
    words = [(m.start(), m.end(), m.group(0)) for m in _WORD_RE.finditer(sentence)]
    spans: list[tuple[int, int, str]] = []
    for index in range(len(words)):
        for length in (2, 3, 4):
            if index + length > len(words):
                break
            chunk = words[index : index + length]
            surface = sentence[chunk[0][0] : chunk[-1][1]]
            if len(surface) != sum(len(w[2]) for w in chunk) + (length - 1):
                continue  # non-space separator (punctuation) inside the span
            texts = [w[2] for w in chunk]
            lowered = [t.casefold() for t in texts]
            if lowered[0] in FUNCTION_WORDS or lowered[-1] in FUNCTION_WORDS:
                continue
            if any(t in FUNCTION_WORDS for t in lowered):
                continue
            if any(len(t) < 2 for t in texts):
                continue
            capitalized = all(t[0].isupper() for t in texts)
            if capitalized and index == 0:
                continue  # sentence-initial capitals are not evidence of an entity
            if not capitalized and length > 3:
                continue  # only entities get to be 4 words long
            spans.append((chunk[0][0], chunk[-1][1], surface))
    return spans


@dataclass(frozen=True)
class PhraseOccurrence:
    """One natural occurrence of a phrase in the corpus."""

    phrase: str
    phrase_id: str
    document_index: int
    document_sha256: str
    char_start: int
    char_end: int
    context_text: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("context_text")
        payload["context_chars"] = len(self.context_text)
        return payload


def mine_phrase_occurrences(
    documents: Sequence[str],
    *,
    config: DatasetConfig,
    phrase_token_ids: Callable[[str], list[int]],
    max_phrases: int | None = None,
) -> tuple[list[PhraseOccurrence], dict]:
    """Mine cohesive phrases and collect natural occurrences of each.

    Args:
        documents: Corpus documents (already filtered for length).
        config: Mining/filtering policy.
        phrase_token_ids: Contextual tokenizer for a phrase — the *same*
            segmentation the verbalizer will emit, so the 2-6 token window is
            enforced on the ids the experiment actually uses.
        max_phrases: Stop after this many accepted phrases (defaults to
            ``config.n_phrases``).

    Returns ``(occurrences, stats)``; ``stats`` records how many candidates each
    filter removed, so a thin dataset is diagnosable without a rerun.
    """
    wanted = int(max_phrases if max_phrases is not None else config.n_phrases)
    counts: dict[str, list[tuple[int, int, int, str]]] = {}
    stats = {
        "documents_scanned": 0,
        "candidate_spans": 0,
        "rejected_function_words": 0,
        "rejected_token_count": 0,
        "rejected_too_few_occurrences": 0,
        "rejected_overlapping_phrase": 0,
        "accepted_phrases": 0,
    }
    for document_index, text in enumerate(documents):
        if document_index >= config.max_documents:
            break
        if len(text.strip()) < config.min_document_chars:
            continue
        stats["documents_scanned"] += 1
        offset = 0
        for sentence in _SENTENCE_SPLIT_RE.split(text):
            for start, end, surface in _candidate_spans(sentence):
                stats["candidate_spans"] += 1
                key = normalize_phrase(surface)
                counts.setdefault(key, []).append(
                    (document_index, offset + start, offset + end, surface)
                )
            offset += len(sentence) + 1

    token_cache: dict[str, int] = {}

    def token_count(surface: str) -> int:
        key = normalize_phrase(surface)
        if key not in token_cache:
            try:
                token_cache[key] = len(phrase_token_ids(surface))
            except AutoencoderError:
                token_cache[key] = -1
        return token_cache[key]

    ordered = sorted(
        counts.items(), key=lambda item: (-len(item[1]), item[0])
    )  # frequent first, ties alphabetical: deterministic across runs
    occurrences: list[PhraseOccurrence] = []
    accepted = 0
    accepted_keys: list[str] = []
    for key, hits in ordered:
        if accepted >= wanted:
            break
        if len(hits) < config.occurrences_per_phrase:
            stats["rejected_too_few_occurrences"] += 1
            continue
        surface = hits[0][3]
        words = [w.casefold() for w in _WORD_RE.findall(surface)]
        if not words or all(w in FUNCTION_WORDS for w in words):
            stats["rejected_function_words"] += 1
            continue
        n_tokens = token_count(surface)
        if not (config.min_phrase_tokens <= n_tokens <= config.max_phrase_tokens):
            stats["rejected_token_count"] += 1
            continue
        # Overlap filter. "Great Barrier", "Barrier Reef", and "Great Barrier
        # Reef" are all mined from the same text; keeping more than one of them
        # would put near-identical concepts in different splits, which is exactly
        # the leakage assert_no_split_leakage exists to catch. The most frequent
        # variant wins (this loop is frequency-ordered).
        if any(key in other or other in key for other in accepted_keys):
            stats["rejected_overlapping_phrase"] += 1
            continue
        accepted_keys.append(key)
        selected = hits[: config.occurrences_per_phrase]
        for document_index, start, end, _surface in selected:
            document = documents[document_index]
            occurrences.append(
                PhraseOccurrence(
                    phrase=surface,
                    phrase_id=phrase_id(surface),
                    document_index=document_index,
                    document_sha256=document_hash(document),
                    char_start=int(start),
                    char_end=int(end),
                    context_text=document[:start],
                )
            )
        accepted += 1
        _ = key
    stats["accepted_phrases"] = accepted
    if accepted == 0:
        raise AutoencoderError(
            f"no phrase survived mining: {stats}. Loosen the token bounds, lower "
            f"occurrences_per_phrase, or supply more documents."
        )
    return occurrences, stats


def context_token_ids(
    tokenizer, text: str, *, max_context_tokens: int, min_context_tokens: int
) -> list[int]:
    """Token ids of the context, keeping the **tail** and one leading BOS.

    The tail is what matters: the captured activation sits at the last context
    position, i.e. the position whose next-token distribution is the phrase's
    first token. Truncating from the right (the tokenizer's default) would throw
    away exactly that.
    """
    encoded = tokenizer(
        text,
        return_tensors=None,
        truncation=False,
        max_length=10**9,
        add_special_tokens=False,
    )
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    ids = [int(t) for t in ids]
    bos = getattr(tokenizer, "bos_token_id", None)
    budget = int(max_context_tokens) - (1 if isinstance(bos, int) else 0)
    tail = ids[-budget:] if budget > 0 else []
    if len(tail) + (1 if isinstance(bos, int) else 0) < int(min_context_tokens):
        raise AutoencoderError(
            f"context has {len(tail)} tokens, fewer than min_context_tokens="
            f"{min_context_tokens}"
        )
    return ([int(bos)] if isinstance(bos, int) else []) + tail


@torch.no_grad()
def capture_source_activation(model, input_ids: torch.Tensor, layer: int) -> torch.Tensor:
    """Layer-``layer`` residual at the **last** position of ``input_ids``.

    Read from the ``block_output`` site through
    :class:`~jlens.hooks.ActivationRecorder` — the same site the lens was fitted
    at and :mod:`jlens.generative` steers. Never from
    ``forward(...).last_hidden_state``, which HuggingFace text models have
    already passed through the final norm.
    """
    with ActivationRecorder(model.layers, at=[layer]) as recorder:
        model.forward(input_ids)
    hidden = recorder.activations[layer]
    # .clone(): indexing returns a view of a buffer the next forward pass owns.
    return hidden[0, -1].detach().float().clone()


@dataclass
class DatasetBuildResult:
    """What a build produced, in memory."""

    records: list[dict]
    activations: torch.Tensor
    cones: torch.Tensor
    phrase_token_ids: list[list[int]]
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)


def build_dataset(
    model,
    lens_dictionary: JSpaceDictionary,
    documents: Sequence[str],
    *,
    dataset_config: DatasetConfig,
    pursuit_config: PursuitConfig,
    phrase_token_ids: Callable[[str], list[int]],
    provenance: dict | None = None,
    progress: Callable[[str], None] | None = None,
    max_phrases: int | None = None,
) -> DatasetBuildResult:
    """Mine, capture, decompose, and record — the whole Phase 2 pipeline.

    Nothing is cached in here; caching is :func:`save_dataset` /
    :func:`load_dataset`, so a caller can always tell whether pursuit ran.
    """
    occurrences, mining_stats = mine_phrase_occurrences(
        documents,
        config=dataset_config,
        phrase_token_ids=phrase_token_ids,
        max_phrases=max_phrases,
    )
    layer = int(dataset_config.source_layer)
    if layer != lens_dictionary.layer:
        raise AutoencoderError(
            f"dataset.source_layer={layer} but the dictionary was built for layer "
            f"{lens_dictionary.layer}"
        )
    device = next(iter(model.layers.parameters())).device

    captured: list[torch.Tensor] = []
    kept: list[PhraseOccurrence] = []
    context_lengths: list[int] = []
    capture_started = time.perf_counter()
    for index, occurrence in enumerate(occurrences):
        try:
            ids = context_token_ids(
                model.tokenizer,
                occurrence.context_text,
                max_context_tokens=dataset_config.max_context_tokens,
                min_context_tokens=dataset_config.min_context_tokens,
            )
        except AutoencoderError:
            continue  # occurrence too close to the start of its document
        tensor = torch.tensor([ids], dtype=torch.long, device=device)
        captured.append(capture_source_activation(model, tensor, layer).cpu())
        kept.append(occurrence)
        context_lengths.append(len(ids))
        if progress is not None and (index + 1) % 25 == 0:
            progress(f"captured {index + 1}/{len(occurrences)} occurrences")
    if not kept:
        raise AutoencoderError(
            "every mined occurrence was rejected for insufficient preceding "
            "context; lower dataset.min_context_tokens"
        )
    capture_seconds = time.perf_counter() - capture_started

    activations = torch.stack(captured)  # [N, d_model]
    settings = PursuitSettings(
        k=pursuit_config.k,
        normalize_atoms=pursuit_config.normalize_atoms,
        refine_steps=pursuit_config.refine_steps,
        tol_relative_residual=pursuit_config.tol_relative_residual,
        correlation_chunk_size=pursuit_config.correlation_chunk_size,
    )
    pursuit_started = time.perf_counter()
    batch = max(1, int(dataset_config.capture_batch_size))
    pursuit_records: list[dict] = []
    cones: list[torch.Tensor] = []
    for start in range(0, activations.shape[0], batch):
        chunk = activations[start : start + batch].to(lens_dictionary.device)
        result = gradient_pursuit(chunk, lens_dictionary, settings)
        for record in result.to_records():
            cone = weighted_reconstruction(
                lens_dictionary.atoms, record["token_ids"], record["coefficients"]
            )
            cones.append(cone.detach().float().cpu())
            pursuit_records.append(record)
        del chunk, result
        if progress is not None:
            progress(f"decomposed {min(start + batch, activations.shape[0])}/{activations.shape[0]}")
    pursuit_seconds = time.perf_counter() - pursuit_started
    cone_tensor = torch.stack(cones)

    splits = assign_splits(
        [occurrence.phrase for occurrence in kept],
        salt=dataset_config.split_salt,
        val_fraction=dataset_config.val_fraction,
        heldout_fraction=dataset_config.heldout_fraction,
    )
    records: list[dict] = []
    token_ids_per_record: list[list[int]] = []
    for index, occurrence in enumerate(kept):
        ids = phrase_token_ids(occurrence.phrase)
        token_ids_per_record.append(ids)
        activation = activations[index]
        cone = cone_tensor[index]
        pursuit_record = pursuit_records[index]
        records.append(
            {
                "schema": DATASET_RECORD_SCHEMA,
                "record_index": index,
                "phrase": occurrence.phrase,
                "phrase_normalized": normalize_phrase(occurrence.phrase),
                "phrase_id": occurrence.phrase_id,
                "phrase_token_ids": list(ids),
                "n_phrase_tokens": len(ids),
                "split": splits[normalize_phrase(occurrence.phrase)],
                "source": occurrence.to_dict(),
                "context_token_len": context_lengths[index],
                "source_layer": layer,
                "source_activation_norm": float(activation.norm()),
                "source_activation_sha256": tensor_sha256(activation),
                "cone_norm": float(cone.norm()),
                "cone_sha256": tensor_sha256(cone),
                "active_token_ids": pursuit_record["token_ids"],
                "active_coefficients": pursuit_record["coefficients"],
                "n_active_atoms": pursuit_record["n_selected"],
                "pursuit_explained_fraction": pursuit_record["explained_fraction"],
                "pursuit_relative_residual": pursuit_record["relative_residual"],
                "pursuit_stop_reason": pursuit_record["stop_reason"],
                "provenance": dict(provenance or {}),
            }
        )

    stats = {
        **mining_stats,
        "n_occurrences_mined": len(occurrences),
        "n_occurrences_captured": len(kept),
        "n_phrases_captured": len({r["phrase_id"] for r in records}),
        "capture_seconds": round(capture_seconds, 3),
        "pursuit_seconds": round(pursuit_seconds, 3),
        "seconds_per_occurrence": round(
            (capture_seconds + pursuit_seconds) / max(1, len(kept)), 4
        ),
        "d_model": int(activations.shape[1]),
        "split_counts": {
            split: sum(1 for r in records if r["split"] == split) for split in SPLITS
        },
        "phrase_split_counts": {
            split: len({r["phrase_id"] for r in records if r["split"] == split})
            for split in SPLITS
        },
    }
    return DatasetBuildResult(
        records=records,
        activations=activations,
        cones=cone_tensor,
        phrase_token_ids=token_ids_per_record,
        stats=stats,
    )


def benchmark_build(
    model,
    lens_dictionary: JSpaceDictionary,
    documents: Sequence[str],
    *,
    dataset_config: DatasetConfig,
    pursuit_config: PursuitConfig,
    phrase_token_ids: Callable[[str], list[int]],
    n_phrases: int = 4,
) -> dict:
    """Run a small slice of the build and project the full cost from it.

    The brief requires measured estimates before the pilot is constructed, so
    this returns *measurements plus an explicit extrapolation*, never a guess:
    per-occurrence seconds and bytes are measured; the projection is linear in
    the number of occurrences and is labelled as such.
    """
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = build_dataset(
        model,
        lens_dictionary,
        documents,
        dataset_config=dataset_config,
        pursuit_config=pursuit_config,
        phrase_token_ids=phrase_token_ids,
        max_phrases=n_phrases,
    )
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    n = len(result)
    d_model = int(result.activations.shape[1])
    bytes_per_occurrence = 2 * d_model * 4 + 1024  # activation + cone (fp32) + record
    planned = dataset_config.n_phrases * dataset_config.occurrences_per_phrase
    return {
        "measured": {
            "n_occurrences": n,
            "wall_seconds": round(elapsed, 3),
            "seconds_per_occurrence": round(elapsed / max(1, n), 4),
            "bytes_per_occurrence": bytes_per_occurrence,
            "peak_cuda_memory_gb": (
                round(torch.cuda.max_memory_allocated() / 2**30, 3) if cuda else None
            ),
            "stats": result.stats,
        },
        "projection": {
            "basis": "linear in occurrence count from the measured slice",
            "planned_occurrences": planned,
            "estimated_wall_minutes": round(elapsed / max(1, n) * planned / 60.0, 2),
            "estimated_storage_mb": round(bytes_per_occurrence * planned / 2**20, 2),
        },
    }


# --------------------------------------------------------------- persistence


def save_dataset(
    directory: str, result: DatasetBuildResult, *, manifest_extra: dict | None = None
) -> dict:
    """Write records, tensors, and a manifest. Tensors are cached so no training
    run ever repeats pursuit."""
    os.makedirs(directory, exist_ok=True)
    records_path = os.path.join(directory, "records.jsonl")
    tmp_records = f"{records_path}.tmp.{os.getpid()}"
    with open(tmp_records, "w", encoding="utf-8") as handle:
        for record in result.records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_records, records_path)

    tensors_path = os.path.join(directory, "tensors.pt")
    tmp_tensors = f"{tensors_path}.tmp.{os.getpid()}"
    torch.save(
        {
            "activations": result.activations,
            "cones": result.cones,
            "phrase_token_ids": result.phrase_token_ids,
        },
        tmp_tensors,
    )
    os.replace(tmp_tensors, tensors_path)

    manifest = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "n_records": len(result.records),
        "n_phrases": len({r["phrase_id"] for r in result.records}),
        "d_model": int(result.activations.shape[1]),
        "stats": result.stats,
        "records_sha256": _file_sha256(records_path),
        "tensors_sha256": _file_sha256(tensors_path),
        **dict(manifest_extra or {}),
    }
    manifest_path = os.path.join(directory, "manifest.json")
    tmp_manifest = f"{manifest_path}.tmp.{os.getpid()}"
    with open(tmp_manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_manifest, manifest_path)
    return manifest


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class JSpaceLanguageDataset:
    """A loaded dataset: records, tensors, splits, and phrase prototypes.

    Tensors are exposed as plain ``[N, d_model]`` matrices aligned with
    ``records`` by index. Nothing here reads a record's provenance strings — the
    training paths take ``cones``, ``phrase_token_ids``, and split membership,
    and nothing else, so metadata physically cannot reach a model.
    """

    def __init__(
        self,
        records: Sequence[dict],
        activations: torch.Tensor,
        cones: torch.Tensor,
        phrase_token_ids: Sequence[Sequence[int]],
        manifest: dict | None = None,
    ) -> None:
        if not (len(records) == activations.shape[0] == cones.shape[0] == len(phrase_token_ids)):
            raise AutoencoderError(
                f"dataset parts disagree on length: {len(records)} records, "
                f"{activations.shape[0]} activations, {cones.shape[0]} cones, "
                f"{len(phrase_token_ids)} token id lists"
            )
        self.records = list(records)
        self.activations = activations
        self.cones = cones
        self.phrase_token_ids = [list(ids) for ids in phrase_token_ids]
        self.manifest = dict(manifest or {})

    def __len__(self) -> int:
        return len(self.records)

    @property
    def d_model(self) -> int:
        return int(self.cones.shape[1])

    @classmethod
    def load(cls, directory: str) -> JSpaceLanguageDataset:
        records_path = os.path.join(directory, "records.jsonl")
        with open(records_path, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        for record in records:
            if record.get("schema") != DATASET_RECORD_SCHEMA:
                raise AutoencoderError(
                    f"{records_path}: unexpected record schema {record.get('schema')!r}"
                )
        payload = torch.load(
            os.path.join(directory, "tensors.pt"), map_location="cpu", weights_only=False
        )
        manifest_path = os.path.join(directory, "manifest.json")
        manifest = {}
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        return cls(
            records,
            payload["activations"],
            payload["cones"],
            payload["phrase_token_ids"],
            manifest,
        )

    @classmethod
    def from_build(cls, result: DatasetBuildResult) -> JSpaceLanguageDataset:
        return cls(
            result.records,
            result.activations,
            result.cones,
            result.phrase_token_ids,
            {"stats": result.stats},
        )

    def indices_for_split(self, split: str) -> list[int]:
        if split not in SPLITS:
            raise AutoencoderError(f"split {split!r} not in {SPLITS}")
        return [i for i, r in enumerate(self.records) if r["split"] == split]

    def phrases_for_split(self, split: str) -> list[str]:
        """Distinct phrase surfaces in a split, in deterministic order."""
        seen: dict[str, str] = {}
        for index in self.indices_for_split(split):
            record = self.records[index]
            seen.setdefault(record["phrase_id"], record["phrase"])
        return [seen[key] for key in sorted(seen)]

    def unit_cones(self) -> torch.Tensor:
        """``cones`` L2-normalized row-wise. Zero rows stay zero (and are
        rejected by :meth:`assert_usable`, so they never reach training)."""
        norms = self.cones.norm(dim=-1, keepdim=True)
        return self.cones / norms.clamp_min(1e-30)

    def prototypes(self, split: str) -> tuple[list[str], torch.Tensor, dict]:
        """Per-phrase prototype cones, built **only** from ``split``'s records.

        The prototype is the re-normalized mean of the phrase's unit cones. It is
        the reconstructor's regression target; deriving it from a single split
        keeps held-out phrases out of the target space entirely.

        Returns ``(phrases, prototypes[P, d], dispersion)`` where ``dispersion``
        reports within-phrase and between-phrase cosine spread — the direct test
        of prototype collapse (risk 2 in the design note).
        """
        indices = self.indices_for_split(split)
        if not indices:
            raise AutoencoderError(f"split {split!r} has no records")
        units = self.unit_cones()
        grouped: dict[str, list[int]] = {}
        surfaces: dict[str, str] = {}
        for index in indices:
            record = self.records[index]
            grouped.setdefault(record["phrase_id"], []).append(index)
            surfaces.setdefault(record["phrase_id"], record["phrase"])
        keys = sorted(grouped)
        rows: list[torch.Tensor] = []
        within: list[float] = []
        for key in keys:
            members = units[grouped[key]]
            mean = members.mean(dim=0)
            norm = float(mean.norm())
            if norm == 0.0:
                raise AutoencoderError(
                    f"phrase {surfaces[key]!r} has cones that cancel to zero; its "
                    f"prototype is undefined"
                )
            prototype = mean / norm
            rows.append(prototype)
            if members.shape[0] > 1:
                within.append(float((members @ prototype).mean()))
        prototypes = torch.stack(rows)
        between = prototypes @ prototypes.T
        off_diagonal = between[~torch.eye(len(keys), dtype=torch.bool)]
        dispersion = {
            "n_phrases": len(keys),
            "mean_within_phrase_cosine": (
                sum(within) / len(within) if within else None
            ),
            "mean_between_phrase_cosine": (
                float(off_diagonal.mean()) if off_diagonal.numel() else None
            ),
            "max_between_phrase_cosine": (
                float(off_diagonal.max()) if off_diagonal.numel() else None
            ),
        }
        return [surfaces[key] for key in keys], prototypes, dispersion

    def assert_usable(self) -> None:
        """Reject a dataset that cannot support the experiment."""
        if len(self) == 0:
            raise AutoencoderError("dataset is empty")
        if not torch.isfinite(self.cones).all():
            raise AutoencoderError("dataset contains non-finite cones")
        zero_rows = int((self.cones.norm(dim=-1) == 0).sum())
        if zero_rows:
            raise AutoencoderError(
                f"{zero_rows} record(s) have a zero cone; the pursuit found no "
                f"positively-correlated atom for them and they carry no signal"
            )
        for split in SPLITS:
            if not self.indices_for_split(split):
                raise AutoencoderError(
                    f"split {split!r} is empty; the dataset cannot support a "
                    f"concept-disjoint evaluation"
                )


def assert_no_split_leakage(dataset: JSpaceLanguageDataset, *, salt: str, val_fraction: float, heldout_fraction: float) -> dict:
    """Re-derive every record's split and fail on any crossing.

    Two independent checks, because they catch different bugs:

    1. **Recomputation** — the stored split must equal the split the salted hash
       of the record's own phrase implies. Catches a shuffled or hand-edited
       assignment.
    2. **String containment** — no normalized held-out phrase may appear inside
       any training phrase's text. Catches the subtler case where "Barrier Reef"
       trains while "Great Barrier Reef" is held out.
    """
    violations: list[dict] = []
    expected_splits = assign_splits(
        [record["phrase"] for record in dataset.records],
        salt=salt,
        val_fraction=val_fraction,
        heldout_fraction=heldout_fraction,
    )
    for record in dataset.records:
        expected = expected_splits[normalize_phrase(record["phrase"])]
        if expected != record["split"]:
            violations.append(
                {
                    "kind": "split_mismatch",
                    "phrase": record["phrase"],
                    "stored": record["split"],
                    "recomputed": expected,
                }
            )
    train_phrases = {normalize_phrase(p) for p in dataset.phrases_for_split("train")}
    heldout_phrases = {normalize_phrase(p) for p in dataset.phrases_for_split("heldout")}
    for heldout in heldout_phrases:
        for train in train_phrases:
            if heldout and (heldout in train or train in heldout):
                violations.append(
                    {"kind": "substring_overlap", "heldout": heldout, "train": train}
                )
    report = {
        "n_records": len(dataset),
        "n_train_phrases": len(train_phrases),
        "n_heldout_phrases": len(heldout_phrases),
        "violations": violations,
        "clean": not violations,
    }
    if violations:
        raise AutoencoderError(
            f"split leakage detected ({len(violations)} violation(s)): "
            f"{violations[:5]}"
        )
    return report


def load_wikitext_documents(
    n_documents: int, *, min_chars: int = 400, split: str = "train"
) -> list[str]:
    """Stream WikiText-103 documents of at least ``min_chars`` characters.

    Thin wrapper over :func:`jlens.examples.load_wikitext_prompts`'s dataset —
    kept separate because the autoencoder wants documents (for phrase mining),
    not fitting prompts, and wants the split to be selectable.
    """
    if n_documents <= 0:
        return []
    from datasets import load_dataset

    stream = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split=split, streaming=True
    )
    documents: list[str] = []
    for record in stream:
        text = record["text"]
        if len(text.strip()) >= min_chars:
            documents.append(text)
            if len(documents) >= n_documents:
                break
    return documents


def iter_batches(items: Sequence[int], batch_size: int) -> Iterable[list[int]]:
    """Deterministic contiguous batching helper."""
    for start in range(0, len(items), int(batch_size)):
        yield list(items[start : start + int(batch_size)])
