# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Importing a completed calibration run as a parent, without touching it.

The early-layer extension continues one specific piece of a completed run: the
**fitting accumulator**, ``{jacobian_sum, n_done}``. Nothing else crosses the
boundary. This module is the border control.

Three jobs, and all three are refusals rather than repairs:

1. **Resolve, do not assume.** :func:`discover_parent_files` walks the parent
   directory and reports what is actually there; :func:`resolve_parent_layout`
   then maps the roles this extension needs onto discovered paths, searching by
   pattern when the canonical name is absent. A missing artifact produces a
   message naming the role, the paths that were tried, and what was found —
   never a guess and never a default.
2. **Prove compatibility before continuing.** :func:`audit_parent_run` checks
   the parent fingerprint, its configuration checksum, the model and tokenizer
   revisions, the corpus identity, the hook site, the layer grid, ``d_model``,
   the estimator, the accumulator's own recorded ``source_layers`` /
   ``target_layer`` / ``skip_first``, and ``n_done == 100``. A single
   disagreement raises :class:`ParentImportRefused`.
3. **Prove immutability afterwards.** :func:`protected_parent_checksums` is
   taken before the extension runs and again after;
   :func:`assert_parent_unchanged` compares them byte-for-byte. Nothing in this
   package opens a parent file for writing, and the check exists to make that
   auditable rather than asserted.

Why the fitting accumulator may be reused and the old confirmation set may not:
``J_l`` is a running mean over a deterministically ordered prompt list, and the
fitting loop never reads a validation or confirmation result — there is no code
path by which a held-out number could have influenced the accumulator. The old
*confirmation* set, by contrast, has been opened and its result is the reason
this extension exists. Reusing it would make the extension's endpoint a set
that has already been consulted, which is exactly what an untouched confirmation
set cannot be. See :mod:`jlens.calibration.extension` for the protocol that
states this and the fresh sets that replace it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jlens.metadata import file_sha256
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "PARENT_OPTIONAL_ROLES",
    "PARENT_PROVENANCE_SCHEMA",
    "PARENT_REQUIRED_ROLES",
    "ParentAccumulator",
    "ParentImportRefused",
    "ParentRequirements",
    "ParentRun",
    "assert_parent_unchanged",
    "audit_parent_run",
    "discover_parent_files",
    "format_parent_audit",
    "load_parent_run",
    "parent_provenance_manifest",
    "protected_parent_checksums",
    "resolve_parent_layout",
]

PARENT_PROVENANCE_SCHEMA = "jlens.calibration.parent_provenance.v1"

#: Roles the extension cannot proceed without, each with the canonical relative
#: path first and the fallback glob patterns that are searched when it is
#: absent. ``{baseline}`` is substituted with the baseline scale (100).
PARENT_REQUIRED_ROLES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "fingerprint",
        ("fingerprint.json",),
        "what the parent's stored results were produced from",
    ),
    (
        "corpus_manifest",
        ("units/corpus/manifest.json", "units/corpus/*.json"),
        "corpus identity, split checksums, and the fit-prefix nesting audit",
    ),
    (
        "fit_summary",
        ("units/fit_diagnostics/summary.json",),
        "n_done, the capture plan, and the snapshot table",
    ),
    (
        "baseline_snapshot_unit",
        ("units/scale_snapshot/scale{baseline}.json", "units/scale_snapshot/*.json"),
        "the scale-{baseline} snapshot's recorded checksum",
    ),
    (
        "accumulator",
        ("checkpoints/jacobian_sum.pt", "checkpoints/*.pt"),
        "the sufficient statistic this extension continues",
    ),
    (
        "baseline_lens",
        ("artifacts/lens.scale{baseline}.pt", "artifacts/lens.scale*.pt"),
        "the scale-{baseline} lens, kept as a descriptive baseline",
    ),
)

#: Roles that are recorded when present and reported as absent when not. None of
#: them gates continuation; all of them are evidence about what the parent did.
PARENT_OPTIONAL_ROLES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "validation",
        ("units/validation/scale{baseline}.json", "units/validation/*.json"),
        "the parent's development verdicts",
    ),
    (
        "confirmation",
        ("units/confirmation/scale{baseline}.json", "units/confirmation/*.json"),
        "the parent's confirmation verdicts — read for provenance, never reused",
    ),
    (
        "publication",
        ("units/publication/scale{baseline}.json", "units/publication/*.json"),
        "what the parent published",
    ),
    (
        "scale_comparison",
        ("units/scale_comparison/comparison.json", "units/scale_comparison/*.json"),
        "the parent's scale table",
    ),
    (
        "report_json",
        ("artifacts/calibration_report.json",),
        "the parent run report, including its confirmation-vault status",
    ),
    (
        "fit_rolling",
        ("units/fit_diagnostics/rolling.json",),
        "rolling per-prompt fit diagnostics",
    ),
)


class ParentImportRefused(RuntimeError):
    """The parent run cannot be proved compatible, so it is not continued.

    Raised rather than worked around. Continuing an accumulator whose provenance
    is unproven would produce a lens whose fit-prompt count is a claim rather
    than a fact.
    """


# --------------------------------------------------------------- requirements


@dataclass(frozen=True)
class ParentRequirements:
    """What the extension requires the parent run to have been.

    Every field is compared against the parent's own recorded metadata. Nothing
    here is defaulted from the parent: a value the extension does not know is a
    value the extension refuses to assume.
    """

    model_repo_id: str
    model_revision: str
    tokenizer_repo_id: str
    tokenizer_revision: str
    source_layers: tuple[int, ...]
    target_layer: int
    d_model: int
    hook_site: str
    skip_first: int
    max_seq_len: int
    dim_batch: int
    corpus_hf_dataset: str
    corpus_config: str
    corpus_split: str
    estimator: str
    artifact_format_version: str
    baseline_scale: int = 100
    expected_n_done: int = 100

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["source_layers"] = [int(layer) for layer in self.source_layers]
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())


@dataclass(frozen=True)
class ParentAccumulator:
    """The parent's fitting checkpoint, read but never written.

    Attributes:
        n_done: Prompts that entered the running mean. This is the number the
            extension continues from, and it must equal the baseline scale.
        next_idx: The parent's resume cursor. Upstream ``fit`` skips prompts
            below it, which is the mechanism the continuation relies on.
    """

    path: str
    checksum: str
    n_done: int
    next_idx: int
    source_layers: tuple[int, ...]
    target_layer: int
    skip_first: int
    layer_shapes: dict[str, list[int]]
    dtype: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["source_layers"] = [int(layer) for layer in self.source_layers]
        return payload


@dataclass(frozen=True)
class ParentRun:
    """A completed calibration run, resolved and read read-only."""

    root: str
    layout: dict
    fingerprint: dict
    fingerprint_digest: str
    corpus_manifest: dict
    fit_summary: dict
    baseline_snapshot: dict
    accumulator: ParentAccumulator
    baseline_lens_checksum: str
    optional: dict = field(default_factory=dict)
    inventory: dict = field(default_factory=dict)

    # ------------------------------------------------------------- accessors

    @property
    def corpus(self) -> dict:
        return dict(self.corpus_manifest.get("corpus", {}))

    @property
    def splits(self) -> dict:
        return dict(self.corpus.get("splits", {}))

    @property
    def split_checksums(self) -> dict:
        return dict(self.splits.get("checksums", {}))

    @property
    def split_sizes(self) -> dict:
        return dict(self.splits.get("sizes", {}))

    @property
    def capture_plan(self) -> dict:
        return dict(self.fit_summary.get("capture_plan", {}))

    @property
    def nesting_audit(self) -> dict:
        return dict(self.corpus_manifest.get("scale_nesting_audit", {}))

    def fit_prefix_checksum(self, scale: int) -> str | None:
        """Checksum over the first ``scale`` fit records, in nested order.

        This is the parent's fit-prompt manifest. It is a checksum over record
        identities — ``record_id``, ``stream_index``, normalized text checksum
        and SimHash — written by the parent's own scale-nesting audit, so the
        extension can prove its reconstruction of the fit ordering reproduces
        the exact prompts the parent fitted on.
        """
        checksums = self.nesting_audit.get("checksums") or {}
        value = checksums.get(str(int(scale)))
        return str(value) if value is not None else None

    @property
    def confirmation_vault_status(self) -> dict:
        report = self.optional.get("report_json") or {}
        return dict(report.get("confirmation_vault") or {})

    @property
    def confirmation_was_opened(self) -> bool | None:
        """Whether the parent opened its confirmation set.

        ``None`` when the parent report is absent. The extension never depends
        on this being ``True``: the old confirmation set is treated as spent
        whichever way this reads, because its *result* is what motivated the
        extension.
        """
        status = self.confirmation_vault_status
        if not status:
            return None
        return bool(status.get("opened"))

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "layout": dict(self.layout),
            "fingerprint": dict(self.fingerprint),
            "fingerprint_digest": self.fingerprint_digest,
            "corpus": self.corpus,
            "splits": self.splits,
            "capture_plan": self.capture_plan,
            "scale_nesting_audit": self.nesting_audit,
            "fit_summary": dict(self.fit_summary),
            "baseline_snapshot": dict(self.baseline_snapshot),
            "accumulator": self.accumulator.to_dict(),
            "baseline_lens_checksum": self.baseline_lens_checksum,
            "optional_present": sorted(self.optional),
            "confirmation_vault_status": self.confirmation_vault_status,
        }


# ------------------------------------------------------------------ discovery


def discover_parent_files(
    root: str | os.PathLike[str], *, max_entries: int = 2_000
) -> dict:
    """Every file under ``root``, with size and modification time.

    Read-only, and deliberately dumb: it reports what is on disk so that a
    refusal can show the operator the actual directory rather than a guess about
    it.
    """
    base = Path(root)
    if not base.is_dir():
        raise ParentImportRefused(
            f"the parent run directory {base} does not exist (or is not a "
            "directory). Mount Drive first, and check the run identifier — this "
            "extension never creates a parent run."
        )
    entries: list[dict] = []
    truncated = False
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if len(entries) >= max_entries:
            truncated = True
            break
        stat = path.stat()
        entries.append(
            {
                "relpath": path.relative_to(base).as_posix(),
                "bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    payload = {
        "root": str(base),
        "n_files": len(entries),
        "truncated": truncated,
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }
    payload["inventory_checksum"] = payload_checksum(payload)
    return payload


def _match_role(
    base: Path, patterns: Sequence[str], *, baseline: int
) -> tuple[str | None, list[str]]:
    """The first existing path for a role, plus every pattern that was tried."""
    tried: list[str] = []
    for pattern in patterns:
        resolved = pattern.format(baseline=int(baseline))
        tried.append(resolved)
        if any(character in resolved for character in "*?["):
            matches = sorted(base.glob(resolved))
            if matches:
                return matches[0].relative_to(base).as_posix(), tried
        elif (base / resolved).is_file():
            return resolved, tried
    return None, tried


def resolve_parent_layout(
    root: str | os.PathLike[str], *, baseline_scale: int = 100
) -> dict:
    """Map the roles the extension needs onto files that actually exist.

    Raises:
        ParentImportRefused: If any required role cannot be resolved. The
            message names each missing role, why it is needed, the paths that
            were searched, and the directories that do exist — so the operator
            knows precisely which artifact has to be recovered.
    """
    base = Path(root)
    inventory = discover_parent_files(base)

    resolved: dict[str, str] = {}
    missing: list[dict] = []
    for role, patterns, why in PARENT_REQUIRED_ROLES:
        relpath, tried = _match_role(base, patterns, baseline=baseline_scale)
        if relpath is None:
            missing.append({"role": role, "why": why.format(baseline=baseline_scale), "tried": tried})
        else:
            resolved[role] = relpath

    optional: dict[str, str] = {}
    absent_optional: list[str] = []
    for role, patterns, _why in PARENT_OPTIONAL_ROLES:
        relpath, _tried = _match_role(base, patterns, baseline=baseline_scale)
        if relpath is None:
            absent_optional.append(role)
        else:
            optional[role] = relpath

    if missing:
        detail = "\n".join(
            f"  - {item['role']}: {item['why']}\n      searched: {item['tried']}"
            for item in missing
        )
        present = "\n".join(f"      {entry['relpath']}" for entry in inventory["files"][:40])
        raise ParentImportRefused(
            f"{base} is missing {len(missing)} required parent artifact(s); "
            "refusing to continue an accumulator whose provenance cannot be "
            "established.\n"
            f"{detail}\n"
            f"  what is actually present ({inventory['n_files']} files, first 40):\n"
            f"{present or '      <empty>'}\n"
            "Recover the named artifact(s) from the parent run, or point this "
            "extension at the run that has them. Nothing is inferred."
        )

    payload = {
        "root": str(base),
        "baseline_scale": int(baseline_scale),
        "required": resolved,
        "optional": optional,
        "optional_absent": absent_optional,
        "inventory_checksum": inventory["inventory_checksum"],
        "n_files": inventory["n_files"],
    }
    payload["layout_checksum"] = payload_checksum(payload)
    return payload


# ------------------------------------------------------------------- loading


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ParentImportRefused(
            f"{path} is not readable JSON ({error}); the parent artifact is torn "
            "or was edited. Recover it from the parent run rather than repairing "
            "it here."
        ) from error


def _read_unit(path: Path, *, fingerprint_digest: str) -> dict:
    """One :class:`~jlens.calibration.state.CalibrationStore` unit's payload.

    The stored ``unit_checksum`` and ``fingerprint_digest`` are both verified.
    A unit that fails either is refused, not skipped: a silently dropped unit is
    how an extension ends up continuing a run it never actually read.
    """
    record = _read_json(path)
    if "payload" not in record:
        raise ParentImportRefused(
            f"{path} does not carry a 'payload' field; it is not a calibration "
            "unit written by jlens.calibration.state.CalibrationStore."
        )
    payload = record["payload"]
    stored = record.get("unit_checksum")
    actual = payload_checksum(payload)
    if stored != actual:
        raise ParentImportRefused(
            f"{path} fails its own checksum (stored {stored}, recomputed "
            f"{actual}); the parent artifact has been modified since it was "
            "written. Refusing to read it."
        )
    bound = record.get("fingerprint_digest")
    if bound != fingerprint_digest:
        raise ParentImportRefused(
            f"{path} is bound to fingerprint {bound}, but the parent run "
            f"directory declares {fingerprint_digest}. The unit and the run "
            "disagree about what produced them."
        )
    return payload


def _load_accumulator(path: Path, *, torch_load=None) -> ParentAccumulator:
    """Read the parent's fitting checkpoint. Opened read-only, never written."""
    if torch_load is None:
        import torch  # noqa: PLC0415 - torch is heavy; only needed for a real parent

        def torch_load(target):
            return torch.load(target, map_location="cpu", weights_only=True)

    try:
        state = torch_load(str(path))
    except Exception as error:  # noqa: BLE001 - reported with the path, not swallowed
        raise ParentImportRefused(
            f"the parent accumulator at {path} could not be loaded ({type(error).__name__}: "
            f"{error}). Refusing to continue a checkpoint that cannot be read."
        ) from error

    for key in ("jacobian_sum", "n_done", "next_idx", "source_layers", "target_layer"):
        if key not in state:
            raise ParentImportRefused(
                f"the parent accumulator at {path} has no {key!r} field; it was "
                "not written by jlens.fitting.fit and its scientific meaning is "
                f"unknown. Present keys: {sorted(state)}."
            )
    jacobian_sum = state["jacobian_sum"]
    shapes = {
        str(layer): [int(dimension) for dimension in tensor.shape]
        for layer, tensor in sorted(jacobian_sum.items())
    }
    dtypes = {str(tensor.dtype) for tensor in jacobian_sum.values()}
    return ParentAccumulator(
        path=str(path),
        checksum=file_sha256(str(path)),
        n_done=int(state["n_done"]),
        next_idx=int(state["next_idx"]),
        source_layers=tuple(int(layer) for layer in state["source_layers"]),
        target_layer=int(state["target_layer"]),
        skip_first=int(state.get("skip_first", -1)),
        layer_shapes=shapes,
        dtype=", ".join(sorted(dtypes)),
    )


def load_parent_run(
    root: str | os.PathLike[str],
    *,
    baseline_scale: int = 100,
    torch_load=None,
) -> ParentRun:
    """Resolve, read and checksum-verify a completed calibration run.

    Nothing is opened for writing and nothing is repaired. Every unit is checked
    against its own checksum and against the run's fingerprint before its
    contents are believed.
    """
    base = Path(root)
    layout = resolve_parent_layout(base, baseline_scale=baseline_scale)
    required = layout["required"]

    fingerprint = _read_json(base / required["fingerprint"])
    fingerprint = {k: v for k, v in fingerprint.items() if k != "written_utc"}
    fingerprint_digest = payload_checksum(fingerprint)

    corpus_manifest = _read_unit(
        base / required["corpus_manifest"], fingerprint_digest=fingerprint_digest
    )
    fit_summary = _read_unit(
        base / required["fit_summary"], fingerprint_digest=fingerprint_digest
    )
    baseline_snapshot = _read_unit(
        base / required["baseline_snapshot_unit"], fingerprint_digest=fingerprint_digest
    )
    accumulator = _load_accumulator(base / required["accumulator"], torch_load=torch_load)
    baseline_lens_checksum = file_sha256(str(base / required["baseline_lens"]))

    optional: dict = {}
    for role, relpath in layout["optional"].items():
        path = base / relpath
        if role == "report_json":
            optional[role] = _read_json(path)
        else:
            optional[role] = _read_unit(path, fingerprint_digest=fingerprint_digest)

    return ParentRun(
        root=str(base),
        layout=layout,
        fingerprint=fingerprint,
        fingerprint_digest=fingerprint_digest,
        corpus_manifest=corpus_manifest,
        fit_summary=fit_summary,
        baseline_snapshot=baseline_snapshot,
        accumulator=accumulator,
        baseline_lens_checksum=baseline_lens_checksum,
        optional=optional,
        inventory={
            "n_files": layout["n_files"],
            "inventory_checksum": layout["inventory_checksum"],
            "layout_checksum": layout["layout_checksum"],
        },
    )


# --------------------------------------------------------------------- audit


def _check(name: str, passed: bool, detail: str, *, blocking: bool = True) -> dict:
    return {
        "check": name,
        "passed": bool(passed),
        "blocking": bool(blocking),
        "detail": detail,
    }


def audit_parent_run(
    parent: ParentRun, *, requirements: ParentRequirements, raise_on_failure: bool = True
) -> dict:
    """Prove — or refuse — that the parent accumulator may be continued.

    Every clause reports its own flag and its own numbers, so a refusal names
    the clause rather than the run.

    Raises:
        ParentImportRefused: If any blocking clause fails and
            ``raise_on_failure`` is set.
    """
    fingerprint = parent.fingerprint
    corpus = parent.corpus
    plan = parent.capture_plan
    accumulator = parent.accumulator
    splits = parent.splits

    plan_layers = tuple(int(layer) for layer in plan.get("layers", ()))
    wanted_layers = tuple(int(layer) for layer in requirements.source_layers)
    baseline = int(requirements.baseline_scale)

    checks: list[dict] = [
        _check(
            "parent_fingerprint_recomputes",
            bool(parent.fingerprint_digest),
            f"parent fingerprint digest {parent.fingerprint_digest} recomputed "
            f"from {parent.layout['required']['fingerprint']}",
        ),
        _check(
            "parent_configuration_checksum",
            corpus.get("corpus_manifest_checksum")
            == fingerprint.get("corpus_manifest_checksum"),
            f"corpus manifest checksum {corpus.get('corpus_manifest_checksum')} "
            f"vs fingerprint {fingerprint.get('corpus_manifest_checksum')}",
        ),
        _check(
            "model_identity",
            fingerprint.get("model_repo_id") == requirements.model_repo_id
            and fingerprint.get("model_revision") == requirements.model_revision,
            f"parent {fingerprint.get('model_repo_id')}@"
            f"{fingerprint.get('model_revision')}; required "
            f"{requirements.model_repo_id}@{requirements.model_revision}",
        ),
        _check(
            "tokenizer_identity",
            fingerprint.get("tokenizer_revision") == requirements.tokenizer_revision,
            f"parent tokenizer revision {fingerprint.get('tokenizer_revision')}; "
            f"required {requirements.tokenizer_revision}",
        ),
        _check(
            "corpus_identity",
            (
                corpus.get("hf_dataset") == requirements.corpus_hf_dataset
                and corpus.get("config") == requirements.corpus_config
                and corpus.get("split") == requirements.corpus_split
            ),
            f"parent {corpus.get('hf_dataset')}/{corpus.get('config')}/"
            f"{corpus.get('split')} rev {corpus.get('revision')} "
            f"({corpus.get('revision_status')}); required "
            f"{requirements.corpus_hf_dataset}/{requirements.corpus_config}/"
            f"{requirements.corpus_split}",
        ),
        _check(
            "corpus_revision_recorded",
            bool(corpus.get("revision")),
            f"parent corpus revision {corpus.get('revision')!r}, status "
            f"{corpus.get('revision_status')!r}",
        ),
        _check(
            "text_only_corpus",
            corpus.get("modality") == "text-only"
            and not corpus.get("spokencoco_used", False)
            and not corpus.get("multimodal_data_used", False),
            f"modality={corpus.get('modality')!r}, "
            f"spokencoco_used={corpus.get('spokencoco_used')}, "
            f"multimodal_data_used={corpus.get('multimodal_data_used')}",
        ),
        _check(
            "hook_site_and_residual_convention",
            plan_layers == wanted_layers
            and int(plan.get("target_layer", -1)) == int(requirements.target_layer)
            and str(fingerprint.get("capture_plan_digest") or ""),
            f"parent layers {list(plan_layers)} -> L{plan.get('target_layer')} at "
            f"{requirements.hook_site}; plan digest "
            f"{fingerprint.get('capture_plan_digest')}",
        ),
        _check(
            "d_model",
            int(plan.get("d_model", -1)) == int(requirements.d_model),
            f"parent d_model {plan.get('d_model')}; required {requirements.d_model}",
        ),
        _check(
            "capture_geometry",
            (
                int(plan.get("skip_first", -1)) == int(requirements.skip_first)
                and int(plan.get("max_seq_len", -1)) == int(requirements.max_seq_len)
                and int(plan.get("dim_batch", -1)) == int(requirements.dim_batch)
            ),
            f"parent skip_first={plan.get('skip_first')}, "
            f"max_seq_len={plan.get('max_seq_len')}, dim_batch={plan.get('dim_batch')}; "
            f"required {requirements.skip_first}/{requirements.max_seq_len}/"
            f"{requirements.dim_batch}",
        ),
        _check(
            "fit_estimator_version",
            parent.fit_summary.get("objective")
            == "not_applicable_estimator_is_a_sample_mean",
            f"parent objective {parent.fit_summary.get('objective')!r}; the "
            f"estimator must be {requirements.estimator}, a sample mean with no "
            "optimizer",
        ),
        _check(
            "artifact_format_version",
            fingerprint.get("artifact_format_version")
            == requirements.artifact_format_version,
            f"parent {fingerprint.get('artifact_format_version')}; required "
            f"{requirements.artifact_format_version}",
        ),
        _check(
            "accumulator_format",
            bool(accumulator.layer_shapes)
            and all(
                shape == [int(requirements.d_model), int(requirements.d_model)]
                for shape in accumulator.layer_shapes.values()
            ),
            f"{len(accumulator.layer_shapes)} layer matrices, shapes "
            f"{sorted({tuple(v) for v in accumulator.layer_shapes.values()})}, "
            f"dtype {accumulator.dtype}",
        ),
        _check(
            "accumulator_layer_grid",
            accumulator.source_layers == wanted_layers
            and accumulator.target_layer == int(requirements.target_layer)
            and accumulator.skip_first == int(requirements.skip_first),
            f"checkpoint source_layers {list(accumulator.source_layers)} -> "
            f"L{accumulator.target_layer}, skip_first={accumulator.skip_first}; "
            f"required {list(wanted_layers)} -> L{requirements.target_layer}, "
            f"skip_first={requirements.skip_first}",
        ),
        _check(
            "accumulator_checksum_recorded",
            accumulator.checksum.startswith("sha256:"),
            f"parent accumulator checksum {accumulator.checksum}",
        ),
        _check(
            "n_done_equals_baseline",
            accumulator.n_done == int(requirements.expected_n_done),
            f"checkpoint n_done={accumulator.n_done} (next_idx="
            f"{accumulator.next_idx}); required {requirements.expected_n_done}",
        ),
        _check(
            "resume_cursor_matches_n_done",
            accumulator.next_idx == accumulator.n_done,
            f"next_idx={accumulator.next_idx}, n_done={accumulator.n_done}; a gap "
            "means the parent skipped prompts and the fit-prompt identity of the "
            "prefix is not recoverable from the checkpoint alone",
        ),
        _check(
            "baseline_snapshot_checksum",
            (
                int(parent.baseline_snapshot.get("n_prompts", -1)) == baseline
                and parent.baseline_snapshot.get("checksum") == parent.baseline_lens_checksum
            ),
            f"snapshot unit n_prompts={parent.baseline_snapshot.get('n_prompts')} "
            f"checksum {parent.baseline_snapshot.get('checksum')}; lens file on "
            f"disk {parent.baseline_lens_checksum}",
        ),
        _check(
            "fit_prompt_ordering_protocol",
            splits.get("protocol") == "stable-hash-bucket-v1"
            and "nested_order" in splits,
            f"split protocol {splits.get('protocol')!r}, fit ordering "
            f"{splits.get('nested_order')!r}",
        ),
        _check(
            "parent_fit_prefix_checksum_present",
            bool(parent.fit_prefix_checksum(baseline)),
            f"scale-{baseline} fit-prefix checksum "
            f"{parent.fit_prefix_checksum(baseline)} (from the parent's own "
            "scale-nesting audit; this is the parent fit-prompt manifest)",
        ),
        _check(
            "no_prompt_dropped_before_fitting",
            int(parent.corpus_manifest.get("n_dropped_too_short", -1)) == 0,
            f"parent dropped {parent.corpus_manifest.get('n_dropped_too_short')} "
            "record(s) as too short before fitting; a non-zero count means the "
            "fitted prefix is not the split prefix and the fitted prompt identity "
            "cannot be reconstructed from stored artifacts",
        ),
        _check(
            "old_split_checksums_present",
            all(
                parent.split_checksums.get(name)
                for name in ("fit", "validation", "confirmation")
            ),
            f"fit={parent.split_checksums.get('fit')}, "
            f"validation={parent.split_checksums.get('validation')}, "
            f"confirmation={parent.split_checksums.get('confirmation')}",
        ),
        _check(
            "old_duplicate_audit_present",
            bool(parent.corpus_manifest.get("leakage_audit", {}).get("audit_checksum")),
            f"parent leakage audit {parent.corpus_manifest.get('leakage_audit', {}).get('audit_checksum')} "
            f"— exact {parent.corpus_manifest.get('leakage_audit', {}).get('n_exact_hits')}, "
            f"near {parent.corpus_manifest.get('leakage_audit', {}).get('n_near_hits')}",
        ),
        _check(
            "old_confirmation_selection_recorded",
            bool(parent.corpus_manifest.get("confirmation_selection_checksum")),
            f"parent confirmation selection checksum "
            f"{parent.corpus_manifest.get('confirmation_selection_checksum')} — "
            "recorded so the old set can be excluded, never so it can be reused",
        ),
        _check(
            "old_confirmation_vault_status_recorded",
            parent.confirmation_was_opened is not None,
            f"parent confirmation vault {parent.confirmation_vault_status or '<no run report>'}",
            blocking=False,
        ),
    ]

    failed = [check["check"] for check in checks if not check["passed"]]
    blocking_failed = [
        check["check"] for check in checks if not check["passed"] and check["blocking"]
    ]
    payload = {
        "schema": "jlens.calibration.parent_audit.v1",
        "parent_root": parent.root,
        "parent_fingerprint_digest": parent.fingerprint_digest,
        "parent_protocol_version": parent.fingerprint.get("protocol_version"),
        "requirements": requirements.to_dict(),
        "requirements_digest": requirements.digest,
        "checks": checks,
        "failed_checks": failed,
        "blocking_failed_checks": blocking_failed,
        "compatible": not blocking_failed,
        "accumulator": accumulator.to_dict(),
        "parent_fit_prefix_checksum": parent.fit_prefix_checksum(baseline),
        "old_split_checksums": parent.split_checksums,
        "old_split_sizes": parent.split_sizes,
        "old_confirmation_selection_checksum": parent.corpus_manifest.get(
            "confirmation_selection_checksum"
        ),
        "old_validation_selection_checksum": (
            parent.corpus_manifest.get("validation_selection") or {}
        ).get("selection_checksum"),
        "old_confirmation_vault_status": parent.confirmation_vault_status,
        "reused_from_parent": ["fitting accumulator (jacobian_sum, n_done)"],
        "never_reused_from_parent": [
            "old development set",
            "old confirmation set",
            "old confirmation verdicts",
        ],
    }
    payload["audit_checksum"] = payload_checksum(payload)

    if blocking_failed and raise_on_failure:
        detail = "\n".join(
            f"  [FAIL] {check['check']}: {check['detail']}"
            for check in checks
            if not check["passed"] and check["blocking"]
        )
        raise ParentImportRefused(
            f"the parent run at {parent.root} cannot be proved compatible with "
            f"this extension; {len(blocking_failed)} blocking check(s) failed.\n"
            f"{detail}\n"
            "Refusing to continue its accumulator. An accumulator whose model, "
            "layer grid, hook site, corpus or prompt count cannot be proved is "
            "not a sufficient statistic for anything."
        )
    return payload


# ----------------------------------------------------------- immutability


def protected_parent_checksums(
    root: str | os.PathLike[str],
    *,
    layout: Mapping | None = None,
    baseline_scale: int = 100,
) -> dict:
    """Checksum every parent file the extension must not change.

    Taken before the extension runs and again afterwards. The set is the
    resolved required and optional roles — the accumulator, the baseline lens
    and every stored unit — because those are the files a careless continuation
    would overwrite.
    """
    base = Path(root)
    layout = layout or resolve_parent_layout(base, baseline_scale=baseline_scale)
    relpaths = sorted(
        {*dict(layout["required"]).values(), *dict(layout["optional"]).values()}
    )
    checksums = {}
    for relpath in relpaths:
        path = base / relpath
        checksums[relpath] = {
            "checksum": file_sha256(str(path)),
            "bytes": int(path.stat().st_size),
        }
    payload = {
        "schema": "jlens.calibration.parent_immutability.v1",
        "root": str(base),
        "n_files": len(checksums),
        "files": checksums,
    }
    payload["checksums_checksum"] = payload_checksum(payload)
    return payload


def assert_parent_unchanged(before: Mapping, after: Mapping) -> dict:
    """Prove the parent run is byte-identical to how the extension found it.

    Raises:
        ParentImportRefused: On any changed, added or removed file. The
            extension writes only inside its own directory; a difference here
            means something else did, and the run's provenance claim is void.
    """
    before_files = dict(before.get("files", {}))
    after_files = dict(after.get("files", {}))
    changed = [
        {
            "relpath": relpath,
            "before": before_files[relpath]["checksum"],
            "after": after_files[relpath]["checksum"],
        }
        for relpath in sorted(set(before_files) & set(after_files))
        if before_files[relpath]["checksum"] != after_files[relpath]["checksum"]
    ]
    removed = sorted(set(before_files) - set(after_files))
    added = sorted(set(after_files) - set(before_files))

    payload = {
        "schema": "jlens.calibration.parent_immutability_proof.v1",
        "root": before.get("root"),
        "n_files_checked": len(before_files),
        "before_checksum": before.get("checksums_checksum"),
        "after_checksum": after.get("checksums_checksum"),
        "changed": changed,
        "removed": removed,
        "added": added,
        "immutable": not (changed or removed or added),
    }
    payload["proof_checksum"] = payload_checksum(payload)
    if not payload["immutable"]:
        raise ParentImportRefused(
            f"the parent run at {before.get('root')} changed while the extension "
            f"ran: {len(changed)} modified, {len(removed)} removed, {len(added)} "
            f"added.\n  modified: {[item['relpath'] for item in changed]}\n"
            f"  removed: {removed}\n  added: {added}\n"
            "This extension never opens a parent file for writing, so this is "
            "not something it can repair. The parent run is the evidence base "
            "for the completed scale-100 study and must be restored."
        )
    return payload


# --------------------------------------------------------------- provenance


def parent_provenance_manifest(
    parent: ParentRun,
    audit: Mapping,
    *,
    immutability: Mapping,
    extension_protocol_version: str,
    extension_run_dir: str,
) -> dict:
    """The read-only record of the parent, written into the *extension* directory.

    Says what was imported, what was deliberately not imported, and why the one
    reusable piece is reusable. Written beside the extension's own artifacts and
    never into the parent run.
    """
    payload = {
        "schema": PARENT_PROVENANCE_SCHEMA,
        "written_by": extension_protocol_version,
        "extension_run_dir": str(extension_run_dir),
        "parent": parent.to_dict(),
        "parent_audit": dict(audit),
        "parent_immutability_before": dict(immutability),
        "read_only": True,
        "parent_written_by_this_extension": False,
        "imported": {
            "fitting_accumulator": {
                "path": parent.accumulator.path,
                "checksum": parent.accumulator.checksum,
                "n_done": parent.accumulator.n_done,
                "why_reusable": (
                    "J_l is a running mean over a deterministically ordered "
                    "prompt list. The fitting loop reads no validation or "
                    "confirmation result, so no held-out number could have "
                    "influenced this accumulator."
                ),
            }
        },
        "not_imported": {
            "old_development_set": (
                "development history; its verdicts are the reason this extension "
                "exists"
            ),
            "old_confirmation_set": (
                "already opened and inspected. Its result motivated this "
                "extension, so it can no longer serve as an untouched endpoint "
                "for a larger scale. It is excluded from every new split rather "
                "than reused, relabelled or reset."
            ),
            "old_confirmation_verdicts": "recorded as history, never re-scored",
            "multimodal_run_results": "no modality other than text participates",
        },
        "descriptive_only": {
            "baseline_scale": parent.baseline_snapshot.get("n_prompts"),
            "baseline_lens_checksum": parent.baseline_lens_checksum,
            "why": (
                "The scale-100 lens is reported as a descriptive baseline. It "
                "cannot be newly confirmed: its confirmation set is spent."
            ),
        },
    }
    payload["provenance_checksum"] = payload_checksum(payload)
    return payload


def format_parent_audit(audit: Mapping) -> str:
    """The block printed by the notebook's parent-audit stage."""
    lines = [
        "PARENT RUN AUDIT — read-only import of one completed calibration",
        f"  parent root        {audit['parent_root']}",
        f"  parent protocol    {audit['parent_protocol_version']}",
        f"  parent fingerprint {audit['parent_fingerprint_digest']}",
        f"  requirements       {audit['requirements_digest']}",
        f"  audit checksum     {audit['audit_checksum']}",
        "",
    ]
    for check in audit["checks"]:
        mark = "pass" if check["passed"] else ("FAIL" if check["blocking"] else "note")
        lines += [f"  [{mark}] {check['check']}", f"         {check['detail']}"]
    accumulator = audit["accumulator"]
    lines += [
        "",
        f"  accumulator        n_done={accumulator['n_done']} "
        f"next_idx={accumulator['next_idx']} layers={accumulator['source_layers']} "
        f"-> L{accumulator['target_layer']}",
        f"  accumulator sha    {accumulator['checksum']}",
        f"  fit-prefix sha     {audit['parent_fit_prefix_checksum']}",
        "",
        f"  REUSED:      {audit['reused_from_parent']}",
        f"  NEVER REUSED: {audit['never_reused_from_parent']}",
        "",
        f"  compatible         {audit['compatible']}",
    ]
    return "\n".join(lines)
