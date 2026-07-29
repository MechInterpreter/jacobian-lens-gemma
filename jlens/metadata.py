# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Experiment configuration and metadata for the Gemma 4 pilot.

Configs are YAML (see ``configs/``). :func:`load_config` validates structure
*before* any model load; :func:`config_fingerprint` gives a stable hash that
binds artifacts to the exact configuration; :func:`environment_manifest` and
:func:`write_metadata` record everything needed to rerun or merge jobs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any

_MODES = ("microsmoke", "smoke", "pilot", "layer21_diagnostic")

#: (dotted key, type(s), required). Validation is structural — values such as
#: layer indices are range-checked later against the actual model by
#: ``jlens.fitting`` / ``jlens.gemma4.verify_architecture``.
_SCHEMA: tuple[tuple[str, tuple[type, ...], bool], ...] = (
    ("mode", (str,), True),
    ("model.repo_id", (str,), True),
    ("model.revision", (str, type(None)), True),
    ("model.tokenizer_repo_id", (str, type(None)), False),
    ("model.tokenizer_revision", (str, type(None)), False),
    ("model.dtype", (str,), True),
    ("model.device_map", (str, type(None)), True),
    ("model.allow_model_load", (bool,), True),
    ("model.expect_n_layers", (int,), True),
    ("model.expect_d_model", (int,), True),
    ("model.expect_vocab_size", (int,), True),
    ("sites.source_site", (str,), True),
    ("sites.target_site", (str,), True),
    ("sites.source_layers", (list,), True),
    ("sites.target_layer", (int,), True),
    ("positions.skip_first", (int,), True),
    ("positions.source_rule", (str,), True),
    ("positions.target_rule", (str,), True),
    ("fitting.prompt_source", (str,), True),
    ("fitting.prompts_path", (str, type(None)), True),
    ("fitting.n_prompts", (int,), True),
    ("fitting.max_seq_len", (int,), True),
    ("fitting.dim_batch", (int,), True),
    ("fitting.seed", (int,), True),
    ("fitting.checkpoint_every", (int, type(None)), True),
    ("eval.prompts_path", (str,), True),
    ("eval.top_k", (int,), True),
    ("eval.control_seed", (int,), True),
    ("paths.checkpoint", (str,), True),
    ("paths.output_dir", (str,), True),
)


def _get_dotted(config: dict, dotted: str) -> tuple[bool, Any]:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def validate_config(config: dict) -> None:
    """Structural validation; raises ``ValueError`` listing every problem."""
    problems: list[str] = []
    for dotted, types, required in _SCHEMA:
        found, value = _get_dotted(config, dotted)
        if not found:
            if required:
                problems.append(f"missing key: {dotted}")
            continue
        if not isinstance(value, types):
            problems.append(
                f"{dotted}: expected {'/'.join(t.__name__ for t in types)}, "
                f"got {type(value).__name__} ({value!r})"
            )
    found, mode = _get_dotted(config, "mode")
    if found and isinstance(mode, str) and mode not in _MODES:
        problems.append(f"mode: {mode!r} not in {_MODES}")
    found, layers = _get_dotted(config, "sites.source_layers")
    if found and isinstance(layers, list):
        if not layers or not all(isinstance(l, int) for l in layers):
            problems.append("sites.source_layers: must be a non-empty list of ints")
    found, source = _get_dotted(config, "fitting.prompt_source")
    if found and source not in ("file", "wikitext"):
        problems.append(f"fitting.prompt_source: {source!r} not in ('file','wikitext')")
    found, skip = _get_dotted(config, "positions.skip_first")
    _, seq = _get_dotted(config, "fitting.max_seq_len")
    if isinstance(skip, int) and isinstance(seq, int) and seq <= skip + 1:
        problems.append(
            f"fitting.max_seq_len={seq} leaves no valid positions with "
            f"positions.skip_first={skip}"
        )
    if problems:
        raise ValueError("invalid config:\n  " + "\n  ".join(problems))


def load_config(path: str) -> dict:
    """Load and validate a YAML experiment config."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    validate_config(config)
    return config


def config_fingerprint(config: dict) -> str:
    """Stable sha256 over the canonical-JSON config (order-independent)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def file_sha256(path: str) -> str:
    """``sha256:<hex>`` fingerprint of a file's bytes (streamed). Used to
    pin artifact files (e.g. a fitted ``lens.pt``) across environments."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


#: Schema for the J-space decomposition workflow config (separate from the
#: fitting schema above; the decomposition consumes a frozen lens and never
#: refits).
_JSPACE_SCHEMA: tuple[tuple[str, tuple[type, ...], bool], ...] = (
    ("mode", (str,), True),
    ("model.repo_id", (str,), True),
    ("model.revision", (str, type(None)), True),
    ("model.dtype", (str,), True),
    ("model.device_map", (str, type(None)), True),
    ("model.allow_model_load", (bool,), True),
    ("model.expect_n_layers", (int,), True),
    ("model.expect_d_model", (int,), True),
    ("model.expect_vocab_size", (int,), True),
    ("lens.run_dir_name", (str,), True),
    ("lens.artifact_relpath", (str,), True),
    ("lens.expect_file_sha256", (str, type(None)), True),
    ("lens.expect_model_revision", (str,), True),
    ("lens.expect_source_layers", (list,), True),
    ("lens.expect_n_prompts", (int,), True),
    ("decomposition.layers", (list,), True),
    ("decomposition.k_values", (list,), True),
    ("decomposition.normalize_atoms", (bool,), True),
    ("decomposition.refine_steps", (int,), True),
    ("decomposition.tol_relative_residual", (float, int), True),
    ("decomposition.fold_final_norm_weight", (bool,), True),
    ("decomposition.correlation_chunk_size", (int, type(None)), True),
    ("eval.prompts_path", (str,), True),
    ("eval.top_k", (int,), True),
    ("eval.control_seed", (int,), True),
    ("eval.max_seq_len", (int,), True),
    ("paths.output_dir", (str,), True),
)


def validate_jspace_config(config: dict) -> None:
    """Structural validation of a J-space decomposition config; raises
    ``ValueError`` listing every problem."""
    problems: list[str] = []
    for dotted, types, required in _JSPACE_SCHEMA:
        found, value = _get_dotted(config, dotted)
        if not found:
            if required:
                problems.append(f"missing key: {dotted}")
            continue
        if not isinstance(value, types):
            problems.append(
                f"{dotted}: expected {'/'.join(t.__name__ for t in types)}, "
                f"got {type(value).__name__} ({value!r})"
            )
    found, mode = _get_dotted(config, "mode")
    if found and mode != "jspace_pursuit":
        problems.append(f"mode: {mode!r} must be 'jspace_pursuit'")
    for key in ("lens.expect_source_layers", "decomposition.layers",
                "decomposition.k_values"):
        found, values = _get_dotted(config, key)
        if found and isinstance(values, list):
            if not values or not all(isinstance(v, int) for v in values):
                problems.append(f"{key}: must be a non-empty list of ints")
    found_layers, layers = _get_dotted(config, "decomposition.layers")
    found_fitted, fitted = _get_dotted(config, "lens.expect_source_layers")
    if (
        found_layers
        and found_fitted
        and isinstance(layers, list)
        and isinstance(fitted, list)
        and not set(layers) <= set(fitted)
    ):
        problems.append(
            f"decomposition.layers {sorted(set(layers) - set(fitted))} not in "
            f"lens.expect_source_layers"
        )
    if problems:
        raise ValueError("invalid jspace config:\n  " + "\n  ".join(problems))


def load_jspace_config(path: str) -> dict:
    """Load and validate a YAML J-space decomposition config."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    validate_jspace_config(config)
    return config


#: Schema for the causal steering smoke-run config (consumes a frozen lens
#: and a completed jspace run's cone records; never refits, never re-decomposes).
_CAUSAL_SCHEMA: tuple[tuple[str, tuple[type, ...], bool], ...] = (
    ("mode", (str,), True),
    ("model.repo_id", (str,), True),
    ("model.revision", (str, type(None)), True),
    ("model.dtype", (str,), True),
    ("model.device_map", (str, type(None)), True),
    ("model.allow_model_load", (bool,), True),
    ("model.expect_n_layers", (int,), True),
    ("model.expect_d_model", (int,), True),
    ("model.expect_vocab_size", (int,), True),
    ("lens.run_dir_name", (str,), True),
    ("lens.artifact_relpath", (str,), True),
    ("lens.expect_file_sha256", (str, type(None)), True),
    ("lens.expect_model_revision", (str,), True),
    ("lens.expect_source_layers", (list,), True),
    ("lens.expect_n_prompts", (int,), True),
    ("source.jspace_run_dir_name", (str,), True),
    ("source.cones_k", (int,), True),
    ("source.expect_config_fingerprint", (str, type(None)), True),
    ("intervention.layers", (list,), True),
    ("intervention.multipliers", (list,), True),
    ("intervention.optional_multipliers", (list,), False),
    ("intervention.families", (list,), True),
    ("intervention.controls", (list,), True),
    ("intervention.norm_preserving", (bool,), True),
    ("intervention.generate_completions", (bool,), True),
    ("intervention.completion_max_new_tokens", (int,), True),
    ("examples.manifest_path", (str,), True),
    ("eval.top_k", (int,), True),
    ("parity.max_abs_logit_diff_tol", (float, int), True),
    ("paths.output_dir", (str,), True),
)


def validate_causal_config(config: dict) -> None:
    """Structural validation of a causal smoke-run config; raises
    ``ValueError`` listing every problem."""
    problems: list[str] = []
    for dotted, types, required in _CAUSAL_SCHEMA:
        found, value = _get_dotted(config, dotted)
        if not found:
            if required:
                problems.append(f"missing key: {dotted}")
            continue
        if not isinstance(value, types):
            problems.append(
                f"{dotted}: expected {'/'.join(t.__name__ for t in types)}, "
                f"got {type(value).__name__} ({value!r})"
            )
    found, mode = _get_dotted(config, "mode")
    if found and mode != "causal_smoke":
        problems.append(f"mode: {mode!r} must be 'causal_smoke'")
    found, layers = _get_dotted(config, "intervention.layers")
    _, fitted = _get_dotted(config, "lens.expect_source_layers")
    if (
        found
        and isinstance(layers, list)
        and isinstance(fitted, list)
        and not set(layers) <= set(fitted)
    ):
        problems.append(
            f"intervention.layers {sorted(set(layers) - set(fitted))} not in "
            f"lens.expect_source_layers"
        )
    found, multipliers = _get_dotted(config, "intervention.multipliers")
    if found and isinstance(multipliers, list):
        if not all(isinstance(m, (int, float)) for m in multipliers):
            problems.append("intervention.multipliers: must be numbers")
        elif 0.0 not in [float(m) for m in multipliers]:
            problems.append(
                "intervention.multipliers must include 0.0 (the parity no-op)"
            )
    if problems:
        raise ValueError("invalid causal config:\n  " + "\n  ".join(problems))


def load_causal_config(path: str) -> dict:
    """Load and validate a YAML causal smoke-run config."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    validate_causal_config(config)
    return config


#: Schema for the multimodal capture config (image/audio-conditioned decoder
#: states probed with the frozen text-fitted lens; exploratory by design).
_MULTIMODAL_SCHEMA: tuple[tuple[str, tuple[type, ...], bool], ...] = (
    ("mode", (str,), True),
    ("model.repo_id", (str,), True),
    ("model.revision", (str, type(None)), True),
    ("model.dtype", (str,), True),
    ("model.device_map", (str, type(None)), True),
    ("model.allow_model_load", (bool,), True),
    ("model.expect_n_layers", (int,), True),
    ("model.expect_d_model", (int,), True),
    ("model.expect_vocab_size", (int,), True),
    ("lens.run_dir_name", (str,), True),
    ("lens.artifact_relpath", (str,), True),
    ("lens.expect_file_sha256", (str, type(None)), True),
    ("lens.expect_model_revision", (str,), True),
    ("lens.expect_source_layers", (list,), True),
    ("lens.expect_n_prompts", (int,), True),
    ("capture.layers", (list,), True),
    ("capture.k", (int,), True),
    ("capture.position", (int,), True),
    ("capture.top_k", (int,), True),
    ("decomposition.normalize_atoms", (bool,), True),
    ("decomposition.refine_steps", (int,), True),
    ("decomposition.tol_relative_residual", (float, int), True),
    ("decomposition.fold_final_norm_weight", (bool,), True),
    ("decomposition.correlation_chunk_size", (int, type(None)), True),
    ("inputs.image_path", (str, type(None)), True),
    ("inputs.image_prompt", (str,), True),
    ("inputs.audio_path", (str, type(None)), True),
    ("inputs.audio_prompt", (str,), True),
    ("paths.output_dir", (str,), True),
)


def validate_multimodal_config(config: dict) -> None:
    """Structural validation of a multimodal capture config; raises
    ``ValueError`` listing every problem."""
    problems: list[str] = []
    for dotted, types, required in _MULTIMODAL_SCHEMA:
        found, value = _get_dotted(config, dotted)
        if not found:
            if required:
                problems.append(f"missing key: {dotted}")
            continue
        if not isinstance(value, types):
            problems.append(
                f"{dotted}: expected {'/'.join(t.__name__ for t in types)}, "
                f"got {type(value).__name__} ({value!r})"
            )
    found, mode = _get_dotted(config, "mode")
    if found and mode != "multimodal_capture":
        problems.append(f"mode: {mode!r} must be 'multimodal_capture'")
    found, layers = _get_dotted(config, "capture.layers")
    _, fitted = _get_dotted(config, "lens.expect_source_layers")
    if (
        found
        and isinstance(layers, list)
        and isinstance(fitted, list)
        and not set(layers) <= set(fitted)
    ):
        problems.append(
            f"capture.layers {sorted(set(layers) - set(fitted))} not in "
            f"lens.expect_source_layers"
        )
    if problems:
        raise ValueError("invalid multimodal config:\n  " + "\n  ".join(problems))


def load_multimodal_config(path: str) -> dict:
    """Load and validate a YAML multimodal capture config."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    validate_multimodal_config(config)
    return config


#: Schema for the generative J-cone steering validation config (frozen lens,
#: fresh per-example pursuit, schedule-weighted steering of neutral
#: verbalization prompts; see jlens.generative and
#: scripts/run_generative_validation.py).
_GENERATIVE_SCHEMA: tuple[tuple[str, tuple[type, ...], bool], ...] = (
    ("mode", (str,), True),
    ("model.repo_id", (str,), True),
    ("model.revision", (str, type(None)), True),
    ("model.dtype", (str,), True),
    ("model.device_map", (str, type(None)), True),
    ("model.allow_model_load", (bool,), True),
    ("model.expect_n_layers", (int,), True),
    ("model.expect_d_model", (int,), True),
    ("model.expect_vocab_size", (int,), True),
    ("lens.run_dir_name", (str,), True),
    ("lens.artifact_relpath", (str,), True),
    ("lens.expect_file_sha256", (str, type(None)), True),
    ("lens.expect_model_revision", (str,), True),
    ("lens.expect_source_layers", (list,), True),
    ("lens.expect_n_prompts", (int,), True),
    ("benchmark.manifest_path", (str,), True),
    ("benchmark.split", (str,), True),
    # Optional; defaults to jlens.generative.MIN/MAX_TARGET_TOKENS.
    ("benchmark.target_token_bounds", (dict,), False),
    ("benchmark.target_token_bounds.min", (int,), False),
    ("benchmark.target_token_bounds.max", (int,), False),
    ("pursuit.k", (int,), True),
    ("pursuit.normalize_atoms", (bool,), True),
    ("pursuit.refine_steps", (int,), True),
    ("pursuit.tol_relative_residual", (float, int), True),
    ("pursuit.correlation_chunk_size", (int, type(None)), True),
    ("pursuit.atoms_dtype", (str,), True),
    ("steering.layers", (list,), True),
    ("steering.strength_ratios", (list,), True),
    ("steering.schedules", (list,), True),
    ("steering.mass_thresholds", (list,), True),
    ("steering.wrong_position_anchor", (int,), True),
    ("steering.conditions", (list,), True),
    ("decode.generate", (bool,), True),
    ("decode.max_new_tokens", (int,), True),
    ("decode.generate_conditions", (list,), True),
    ("decode.ratios", (list,), True),
    ("decode.schedule_kinds", (list,), True),
    ("neutral_prompts", (list,), True),
    ("eval.control_seed", (int,), True),
    ("parity.max_abs_logit_diff_tol", (float, int), True),
    # Optional greedy-equivalence gate thresholds. The runner supplies defaults:
    # greedy_top_k = jlens.generative.DEFAULT_AGREEMENT_TOP_K, the top-k log-prob
    # tolerance falls back to max_abs_logit_diff_tol, total variation to 0.02.
    ("parity.greedy_top_k", (int,), False),
    ("parity.max_abs_logprob_diff_topk_tol", (float, int), False),
    ("parity.max_total_variation_tol", (float, int), False),
    ("paths.output_dir", (str,), True),
)


def validate_generative_config(config: dict) -> None:
    """Structural validation of a generative validation config; raises
    ``ValueError`` listing every problem."""
    problems: list[str] = []
    for dotted, types, required in _GENERATIVE_SCHEMA:
        found, value = _get_dotted(config, dotted)
        if not found:
            if required:
                problems.append(f"missing key: {dotted}")
            continue
        if not isinstance(value, types):
            problems.append(
                f"{dotted}: expected {'/'.join(t.__name__ for t in types)}, "
                f"got {type(value).__name__} ({value!r})"
            )
    found, mode = _get_dotted(config, "mode")
    if found and mode != "generative_validation":
        problems.append(f"mode: {mode!r} must be 'generative_validation'")
    found, split = _get_dotted(config, "benchmark.split")
    if found and split not in ("dev", "heldout"):
        problems.append(f"benchmark.split: {split!r} not in ('dev', 'heldout')")
    found, layers = _get_dotted(config, "steering.layers")
    _, fitted = _get_dotted(config, "lens.expect_source_layers")
    if (
        found
        and isinstance(layers, list)
        and isinstance(fitted, list)
        and not set(layers) <= set(fitted)
    ):
        problems.append(
            f"steering.layers {sorted(set(layers) - set(fitted))} not in "
            f"lens.expect_source_layers"
        )
    if found and isinstance(layers, list) and len(layers) < 2:
        problems.append(
            "steering.layers needs at least 2 layers (the wrong_layer control "
            "borrows the farthest other steering layer)"
        )
    found, ratios = _get_dotted(config, "steering.strength_ratios")
    if found and isinstance(ratios, list):
        if not ratios or not all(
            isinstance(r, (int, float)) and r > 0 for r in ratios
        ):
            problems.append("steering.strength_ratios: must be positive numbers")
    found, conditions = _get_dotted(config, "steering.conditions")
    if found and isinstance(conditions, list):
        from jlens.generative import VECTOR_CONDITIONS

        unknown = [c for c in conditions if c not in VECTOR_CONDITIONS]
        if unknown:
            problems.append(
                f"steering.conditions: unknown conditions {unknown}"
            )
        for gate in ("none", "zero"):
            if gate not in conditions:
                problems.append(
                    f"steering.conditions must include {gate!r} (baseline/parity)"
                )
    found, schedules = _get_dotted(config, "steering.schedules")
    if found and isinstance(schedules, list):
        from jlens.generative import SCHEDULE_KINDS

        for entry in schedules:
            if not isinstance(entry, dict) or entry.get("kind") not in SCHEDULE_KINDS:
                problems.append(
                    f"steering.schedules: each entry needs kind in "
                    f"{SCHEDULE_KINDS}, got {entry!r}"
                )
    if problems:
        raise ValueError("invalid generative config:\n  " + "\n  ".join(problems))


def load_generative_config(path: str) -> dict:
    """Load and validate a YAML generative validation config."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    validate_generative_config(config)
    return config


def execution_record(
    *,
    configured_allow_model_load: bool,
    resolved_allow_model_load: bool,
    model_loaded: bool,
    override_source: str | None = None,
) -> dict:
    """Provenance block distinguishing static configuration from what a run
    actually did.

    The completed jspace run recorded the static notebook YAML verbatim
    (``config.model.allow_model_load: false``) while its ``load_info`` block
    shows Gemma *was* loaded — the config value was overridden at execution
    time but the resolved value was never written down. This record makes
    the three facts explicit and self-consistent:

    - ``configured_allow_model_load``: the static config/YAML value;
    - ``resolved_allow_model_load``: the value in effect after CLI flags or
      notebook overrides (``override_source`` says where the override came
      from, e.g. ``"cli:--allow-model-load"`` or ``"notebook"``);
    - ``model_loaded``: whether a real model load actually happened.

    Raises ``ValueError`` for the impossible combination (model loaded
    while loading was resolved off), so a run can never write a record with
    the old ambiguity. Existing run artifacts are historical facts and are
    not rewritten; new runs should store this under an ``"execution"`` key
    next to the static config.
    """
    if model_loaded and not resolved_allow_model_load:
        raise ValueError(
            "inconsistent execution record: model_loaded=True but "
            "resolved_allow_model_load=False"
        )
    if (
        override_source is None
        and resolved_allow_model_load != configured_allow_model_load
    ):
        raise ValueError(
            "resolved_allow_model_load differs from the configured value; "
            "record where the override came from via override_source"
        )
    return {
        "schema": "jlens.metadata.execution.v1",
        "configured_allow_model_load": bool(configured_allow_model_load),
        "resolved_allow_model_load": bool(resolved_allow_model_load),
        "override_source": override_source,
        "model_loaded": bool(model_loaded),
    }


def prompt_hashes(prompts: list[str]) -> list[str]:
    """Stable short hash per prompt, so prompt sets are reproducible without
    committing full text."""
    return [hashlib.sha256(p.encode()).hexdigest()[:16] for p in prompts]


def local_git_commit(repo_dir: str | None = None) -> str | None:
    """HEAD commit of the local source tree (None outside a git checkout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and len(sha) == 40 else None


UPSTREAM_REPO_URL = "https://github.com/anthropics/jacobian-lens"
UPSTREAM_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"


def environment_manifest() -> dict:
    """Versions and platform facts recorded with every artifact."""
    import torch
    import transformers

    manifest = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "upstream_repo": UPSTREAM_REPO_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "local_commit": local_git_commit(),
    }
    return manifest


def write_metadata(path: str, payload: dict) -> None:
    """Write ``payload`` as pretty JSON, adding a UTC timestamp; parents are
    created. Uses atomic replace so a crash never leaves a torn file."""
    payload = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **payload,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)
