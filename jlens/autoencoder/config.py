# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Typed configuration for the J-space language autoencoder.

Frozen dataclasses rather than free-form dicts: every field has a declared type
and a validated range, unknown keys are rejected (a typo in a YAML key is a
silent behaviour change otherwise), and :meth:`AutoencoderConfig.to_dict` round
-trips exactly what was used so a run's ``resolved_config.json`` is the config,
not a description of it.

Loading validates **before** any model, corpus, or lens is touched, in the same
spirit as :func:`jlens.metadata.load_config`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypeVar

from jlens.autoencoder.errors import AutoencoderError

#: Config ``mode`` this package accepts. A generative-validation config loaded
#: here (or vice versa) fails immediately instead of half-working.
AUTOENCODER_MODE = "jspace_language_autoencoder"

#: Dataset build sizes. ``smoke`` is the CPU/mock path; ``pilot`` is the real
#: L4 run. ``mock`` additionally forces the deterministic in-process corpus.
DATASET_MODES = ("smoke", "pilot")

CORPUS_KINDS = ("wikitext", "mock")

SPLITS = ("train", "val", "heldout")

T = TypeVar("T")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AutoencoderError(message)


def _from_dict(cls: type[T], payload: Any, *, where: str) -> T:
    """Build a dataclass from a mapping, rejecting unknown and mistyped keys.

    Unknown keys are an error rather than a warning: a config that silently
    ignores ``recontructor.hidden_dim`` trains a different model than the one
    the author wrote down, and nothing downstream would ever notice.
    """
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise AutoencoderError(f"{where}: expected a mapping, got {type(payload).__name__}")
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(payload) - set(known))
    if unknown:
        raise AutoencoderError(
            f"{where}: unknown key(s) {unknown}; known keys are {sorted(known)}"
        )
    kwargs: dict[str, Any] = {}
    for name, value in payload.items():
        spec = known[name]
        kwargs[name] = _coerce(value, spec.type, where=f"{where}.{name}")
    return cls(**kwargs)  # type: ignore[call-arg]


def _coerce(value: Any, annotation: Any, *, where: str) -> Any:
    """Minimal, explicit coercion for the annotations this module uses.

    Only the forms that actually appear here are handled (``int``, ``float``,
    ``bool``, ``str``, ``str | None``, ``list[...]``, ``tuple[...]``); anything
    else is passed through untouched so an unhandled annotation cannot be
    silently mangled.
    """
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    if text.startswith("list[") or text == "list":
        _require(isinstance(value, (list, tuple)), f"{where}: expected a list")
        return list(value)
    if text.startswith("tuple["):
        _require(isinstance(value, (list, tuple)), f"{where}: expected a list")
        return tuple(value)
    if text in ("int",):
        _require(
            isinstance(value, int) and not isinstance(value, bool),
            f"{where}: expected an int, got {value!r}",
        )
        return int(value)
    if text in ("float",):
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{where}: expected a number, got {value!r}",
        )
        return float(value)
    if text in ("bool",):
        _require(isinstance(value, bool), f"{where}: expected a bool, got {value!r}")
        return bool(value)
    if text in ("str",):
        _require(isinstance(value, str), f"{where}: expected a string, got {value!r}")
        return str(value)
    if text in ("str | None", "None | str"):
        _require(
            value is None or isinstance(value, str), f"{where}: expected a string or null"
        )
        return value
    if text in ("int | None", "None | int"):
        _require(
            value is None or (isinstance(value, int) and not isinstance(value, bool)),
            f"{where}: expected an int or null",
        )
        return value
    return value


@dataclass(frozen=True)
class ModelConfig:
    """The frozen Gemma checkpoint and how it is loaded."""

    repo_id: str = "google/gemma-4-E4B-it"
    revision: str | None = None
    dtype: str = "bfloat16"
    device_map: str | None = None
    allow_model_load: bool = False
    expect_n_layers: int = 42
    expect_d_model: int = 2560
    expect_vocab_size: int = 262144

    def validate(self) -> None:
        _require(bool(self.repo_id), "model.repo_id must be non-empty")
        _require(
            self.dtype in ("bfloat16", "float16", "float32"),
            f"model.dtype {self.dtype!r} not in ('bfloat16','float16','float32')",
        )
        _require(self.expect_n_layers > 0, "model.expect_n_layers must be > 0")
        _require(self.expect_d_model > 0, "model.expect_d_model must be > 0")
        _require(self.expect_vocab_size > 0, "model.expect_vocab_size must be > 0")


@dataclass(frozen=True)
class LensConfig:
    """The frozen fitted lens the J-space dictionary is built from."""

    run_dir_name: str = ""
    artifact_relpath: str = "artifacts/lens.pt"
    expect_file_sha256: str | None = None
    expect_model_revision: str | None = None
    expect_source_layers: list[int] = field(default_factory=lambda: [14])

    def validate(self, source_layer: int) -> None:
        _require(
            source_layer in list(self.expect_source_layers),
            f"lens.expect_source_layers {list(self.expect_source_layers)} does not "
            f"contain the configured source layer {source_layer}",
        )


@dataclass(frozen=True)
class PursuitConfig:
    """Sparse decomposition settings. ``k`` is fixed at 10 for this study."""

    k: int = 10
    normalize_atoms: bool = True
    refine_steps: int = 2
    tol_relative_residual: float = 0.0
    correlation_chunk_size: int | None = 65536
    atoms_dtype: str = "float32"
    build_chunk_rows: int | None = 16384

    def validate(self) -> None:
        _require(self.k >= 1, f"pursuit.k must be >= 1, got {self.k}")
        _require(
            self.atoms_dtype in ("float32", "bfloat16", "float16"),
            f"pursuit.atoms_dtype {self.atoms_dtype!r} unsupported",
        )
        _require(
            0.0 <= self.tol_relative_residual < 1.0,
            "pursuit.tol_relative_residual must be in [0, 1)",
        )


@dataclass(frozen=True)
class DatasetConfig:
    """Corpus mining, capture, and split policy."""

    mode: str = "smoke"
    corpus: str = "wikitext"
    source_layer: int = 14
    n_phrases: int = 32
    occurrences_per_phrase: int = 2
    min_phrase_tokens: int = 2
    max_phrase_tokens: int = 6
    min_context_tokens: int = 8
    max_context_tokens: int = 128
    max_documents: int = 2000
    min_document_chars: int = 400
    val_fraction: float = 0.2
    heldout_fraction: float = 0.2
    split_salt: str = "jspace-language-autoencoder-v1"
    seed: int = 1234
    capture_batch_size: int = 8
    benchmark_batches: int = 1

    def validate(self) -> None:
        _require(self.mode in DATASET_MODES, f"dataset.mode {self.mode!r} not in {DATASET_MODES}")
        _require(self.corpus in CORPUS_KINDS, f"dataset.corpus {self.corpus!r} not in {CORPUS_KINDS}")
        _require(self.source_layer >= 0, "dataset.source_layer must be >= 0")
        _require(self.n_phrases >= 4, "dataset.n_phrases must be >= 4 (splits need members)")
        _require(self.occurrences_per_phrase >= 1, "dataset.occurrences_per_phrase must be >= 1")
        _require(
            2 <= self.min_phrase_tokens <= self.max_phrase_tokens <= 6,
            "dataset phrase token bounds must satisfy 2 <= min <= max <= 6",
        )
        _require(self.min_context_tokens >= 1, "dataset.min_context_tokens must be >= 1")
        _require(
            self.max_context_tokens > self.min_context_tokens,
            "dataset.max_context_tokens must exceed min_context_tokens",
        )
        _require(
            0.0 < self.val_fraction < 1.0 and 0.0 < self.heldout_fraction < 1.0,
            "dataset split fractions must be in (0, 1)",
        )
        _require(
            self.val_fraction + self.heldout_fraction < 0.9,
            "dataset.val_fraction + heldout_fraction must leave a training majority",
        )
        _require(self.capture_batch_size >= 1, "dataset.capture_batch_size must be >= 1")


@dataclass(frozen=True)
class ReconstructorConfig:
    """The phrase → q_hat network and its (independent) training."""

    hidden_dim: int = 512
    n_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.1
    max_phrase_tokens: int = 8
    use_source_layer_embedding: bool = True
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 17
    contrastive_weight: float = 1.0
    temperature: float = 0.07
    n_distractors: int = 16
    gate_auroc_min: float = 0.80
    gate_top5_min: float = 0.50

    def validate(self) -> None:
        _require(self.hidden_dim >= 8, "reconstructor.hidden_dim must be >= 8")
        _require(self.contrastive_weight >= 0, "reconstructor.contrastive_weight must be >= 0")
        _require(self.temperature > 0, "reconstructor.temperature must be > 0")
        _require(
            self.hidden_dim % self.n_heads == 0,
            f"reconstructor.hidden_dim {self.hidden_dim} must be divisible by "
            f"n_heads {self.n_heads}",
        )
        _require(self.n_layers >= 1, "reconstructor.n_layers must be >= 1")
        _require(0.0 <= self.dropout < 1.0, "reconstructor.dropout must be in [0, 1)")
        _require(self.epochs >= 1, "reconstructor.epochs must be >= 1")
        _require(self.batch_size >= 1, "reconstructor.batch_size must be >= 1")
        _require(self.learning_rate > 0, "reconstructor.learning_rate must be > 0")
        _require(self.n_distractors >= 1, "reconstructor.n_distractors must be >= 1")
        _require(
            0.0 <= self.gate_auroc_min <= 1.0 and 0.0 <= self.gate_top5_min <= 1.0,
            "reconstructor gate thresholds must be in [0, 1]",
        )


@dataclass(frozen=True)
class AdapterConfig:
    """The q → memory adapter and its supervised warm start."""

    n_memory_tokens: int = 4
    hidden_dim: int = 1024
    dropout: float = 0.0
    memory_rms_scale: float = 1.0
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 23
    max_new_tokens: int = 8
    beam_width: int = 8

    def validate(self) -> None:
        _require(1 <= self.n_memory_tokens <= 32, "adapter.n_memory_tokens must be in [1, 32]")
        _require(self.hidden_dim >= 8, "adapter.hidden_dim must be >= 8")
        _require(0.0 <= self.dropout < 1.0, "adapter.dropout must be in [0, 1)")
        _require(self.memory_rms_scale > 0, "adapter.memory_rms_scale must be > 0")
        _require(self.epochs >= 1, "adapter.epochs must be >= 1")
        _require(self.batch_size >= 1, "adapter.batch_size must be >= 1")
        _require(self.learning_rate > 0, "adapter.learning_rate must be > 0")
        _require(
            1 <= self.max_new_tokens <= 8,
            "adapter.max_new_tokens must be in [1, 8] (the brief fixes the cap at 8)",
        )
        _require(self.beam_width >= 1, "adapter.beam_width must be >= 1")


@dataclass(frozen=True)
class PolicyGradientConfig:
    """Optional REINFORCE refinement. Disabled by default, on purpose."""

    enabled: bool = False
    epochs: int = 2
    learning_rate: float = 1e-5
    baseline: str = "mean"

    def validate(self) -> None:
        _require(self.baseline in ("mean", "none"), "policy_gradient.baseline must be 'mean' or 'none'")
        _require(self.epochs >= 1, "policy_gradient.epochs must be >= 1")


@dataclass(frozen=True)
class PreferenceConfig:
    """Reconstructor-guided preference optimization of the adapter."""

    epochs: int = 4
    batch_size: int = 4
    learning_rate: float = 2e-4
    beta: float = 0.1
    reward_gap: float = 0.02
    max_pairs_per_example: int = 4
    weight_reconstruction: float = 1.0
    weight_margin: float = 1.0
    brevity_penalty: float = 0.02
    brevity_target_tokens: int = 4
    duplicate_penalty: float = 0.1
    n_unrelated_cones: int = 8
    length_normalize: bool = True
    seed: int = 29
    policy_gradient: PolicyGradientConfig = field(default_factory=PolicyGradientConfig)

    def validate(self) -> None:
        _require(self.epochs >= 1, "preference.epochs must be >= 1")
        _require(self.batch_size >= 1, "preference.batch_size must be >= 1")
        _require(self.learning_rate > 0, "preference.learning_rate must be > 0")
        _require(self.beta > 0, "preference.beta must be > 0")
        _require(self.reward_gap >= 0, "preference.reward_gap must be >= 0")
        _require(self.max_pairs_per_example >= 1, "preference.max_pairs_per_example must be >= 1")
        _require(self.n_unrelated_cones >= 1, "preference.n_unrelated_cones must be >= 1")
        _require(self.brevity_penalty >= 0, "preference.brevity_penalty must be >= 0")
        _require(self.duplicate_penalty >= 0, "preference.duplicate_penalty must be >= 0")
        self.policy_gradient.validate()


@dataclass(frozen=True)
class EvaluationConfig:
    """Held-out evaluation, acceptance policy, and GO/NO-GO thresholds."""

    beam_width: int = 8
    max_new_tokens: int = 8
    n_unrelated_cones: int = 8
    n_distractors: int = 16
    accept_recon_min: float = 0.30
    accept_margin_min: float = 0.05
    paraphrase_prompt_ids: list[str] = field(
        default_factory=lambda: ["verbalizer-default", "verbalizer-paraphrase-a", "verbalizer-paraphrase-b"]
    )
    confabulation_attractors: list[str] = field(
        default_factory=lambda: [
            "black hole",
            "photosynthesis",
            "quantum entanglement",
            "Great Barrier Reef",
        ]
    )
    gate_auroc_min: float = 0.80
    gate_top5_min: float = 0.50
    gate_beam_gain_min: float = 0.10
    gate_precision_gain_min: float = 0.10
    gate_control_acceptance_ratio_max: float = 0.5
    seed: int = 31

    def validate(self) -> None:
        _require(self.beam_width >= 1, "evaluation.beam_width must be >= 1")
        _require(
            1 <= self.max_new_tokens <= 8,
            "evaluation.max_new_tokens must be in [1, 8]",
        )
        _require(self.n_unrelated_cones >= 1, "evaluation.n_unrelated_cones must be >= 1")
        _require(self.n_distractors >= 1, "evaluation.n_distractors must be >= 1")
        _require(
            bool(self.paraphrase_prompt_ids), "evaluation.paraphrase_prompt_ids must be non-empty"
        )
        _require(
            bool(self.confabulation_attractors),
            "evaluation.confabulation_attractors must be non-empty",
        )


@dataclass(frozen=True)
class PathsConfig:
    output_dir: str = "artifacts/jspace_language_autoencoder"

    def validate(self) -> None:
        _require(bool(self.output_dir), "paths.output_dir must be non-empty")


@dataclass(frozen=True)
class AutoencoderConfig:
    """The whole experiment configuration."""

    mode: str = AUTOENCODER_MODE
    model: ModelConfig = field(default_factory=ModelConfig)
    lens: LensConfig = field(default_factory=LensConfig)
    pursuit: PursuitConfig = field(default_factory=PursuitConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reconstructor: ReconstructorConfig = field(default_factory=ReconstructorConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def validate(self) -> None:
        _require(
            self.mode == AUTOENCODER_MODE,
            f"mode {self.mode!r} must be {AUTOENCODER_MODE!r}",
        )
        self.model.validate()
        self.pursuit.validate()
        self.dataset.validate()
        self.lens.validate(self.dataset.source_layer)
        self.reconstructor.validate()
        self.adapter.validate()
        self.preference.validate()
        self.evaluation.validate()
        self.paths.validate()
        _require(
            self.reconstructor.max_phrase_tokens >= self.dataset.max_phrase_tokens,
            f"reconstructor.max_phrase_tokens ({self.reconstructor.max_phrase_tokens}) "
            f"must cover dataset.max_phrase_tokens ({self.dataset.max_phrase_tokens})",
        )
        _require(
            self.adapter.max_new_tokens >= self.dataset.max_phrase_tokens,
            f"adapter.max_new_tokens ({self.adapter.max_new_tokens}) must be able to "
            f"emit a full {self.dataset.max_phrase_tokens}-token phrase",
        )
        _require(
            self.evaluation.gate_auroc_min >= self.reconstructor.gate_auroc_min,
            "evaluation.gate_auroc_min must not be laxer than the reconstructor gate",
        )

    @classmethod
    def from_dict(cls, payload: dict) -> AutoencoderConfig:
        if not isinstance(payload, dict):
            raise AutoencoderError("config: top level must be a mapping")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise AutoencoderError(
                f"config: unknown top-level key(s) {unknown}; known keys are {sorted(known)}"
            )
        preference_payload = dict(payload.get("preference") or {})
        policy_payload = preference_payload.pop("policy_gradient", None)
        preference = _from_dict(PreferenceConfig, preference_payload, where="preference")
        if policy_payload is not None:
            preference = PreferenceConfig(
                **{
                    **{f.name: getattr(preference, f.name) for f in fields(PreferenceConfig)},
                    "policy_gradient": _from_dict(
                        PolicyGradientConfig, policy_payload, where="preference.policy_gradient"
                    ),
                }
            )
        config = cls(
            mode=str(payload.get("mode", AUTOENCODER_MODE)),
            model=_from_dict(ModelConfig, payload.get("model"), where="model"),
            lens=_from_dict(LensConfig, payload.get("lens"), where="lens"),
            pursuit=_from_dict(PursuitConfig, payload.get("pursuit"), where="pursuit"),
            dataset=_from_dict(DatasetConfig, payload.get("dataset"), where="dataset"),
            reconstructor=_from_dict(
                ReconstructorConfig, payload.get("reconstructor"), where="reconstructor"
            ),
            adapter=_from_dict(AdapterConfig, payload.get("adapter"), where="adapter"),
            preference=preference,
            evaluation=_from_dict(EvaluationConfig, payload.get("evaluation"), where="evaluation"),
            paths=_from_dict(PathsConfig, payload.get("paths"), where="paths"),
        )
        config.validate()
        return config

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable sha256 over the canonical JSON of the resolved config."""
        import hashlib

        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def load_autoencoder_config(path: str) -> AutoencoderConfig:
    """Load and validate a YAML autoencoder config."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise AutoencoderError(f"{path}: top level must be a mapping")
    return AutoencoderConfig.from_dict(payload)
