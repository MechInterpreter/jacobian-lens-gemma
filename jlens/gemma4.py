# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Gemma 4 (E4B) adapter for the Jacobian lens.

This module is the narrow, Gemma-specific layer on top of the upstream
model-agnostic package. It does four things:

1. **Revision pinning** — :func:`resolve_revision` turns a mutable ref
   (``main``) into an immutable HuggingFace commit SHA and fails clearly if it
   cannot.
2. **Gated loading** — :func:`load_gemma4` loads the multimodal
   ``Gemma4ForConditionalGeneration`` checkpoint (towers included; text-only
   execution never touches them) at the pinned revision. Loading is refused
   unless the caller passes ``allow_model_load=True`` explicitly.
3. **Explicit tokenization control** — :class:`Gemma4LensModel` subclasses the
   upstream :class:`~jlens.hf.HFLensModel` to (a) prepend BOS deterministically
   in :meth:`~Gemma4LensModel.encode` instead of relying on tokenizer
   attributes, and (b) expose pre-softcap logits next to Gemma's softcapped
   output pathway (:meth:`~Gemma4LensModel.unembed_pair`).
4. **Architecture verification** — :func:`verify_architecture` checks the
   assumptions the lens depends on and returns a serialisable report.
   It hard-fails only on genuine incompatibilities (MoE routing enabled,
   unexpected width/depth, trainable parameters); soft facts such as
   ``layer_scalar`` values or tokenizer BOS behaviour are *recorded*, not
   asserted.

Residual sites (Gemma 4 E4B text decoder, 42 blocks, d_model 2560):

- **Source** ``h_l``: the tensor returned by
  ``model.language_model.layers[l]`` — the residual stream *after* attention,
  MLP, per-layer-embedding (PLE) re-injection, and the block's
  ``layer_scalar``; i.e. the exact input to block ``l+1``.
- **Target** ``h_final``: the same site at the last block (index 41), the
  pre-final-norm residual.

Readout: ``lm_head(norm(J_l h))`` where ``norm`` is the final RMSNorm and
``lm_head`` the tied ``[vocab, d_model]`` unembedding, then (for Gemma's own
pathway) the ``30 * tanh(x / 30)`` final-logit softcap. The softcap is
strictly monotonic, so token *rankings* are identical pre/post cap; only
logit values and softmax probabilities differ.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

from jlens.hf import HFLensModel, Layout
from jlens.lens import JacobianLens

logger = logging.getLogger(__name__)

GEMMA4_E4B_REPO = "google/gemma-4-E4B-it"

#: Layout of ``Gemma4ForConditionalGeneration``: text decoder at
#: ``model.language_model`` (``layers`` / ``norm`` / ``embed_tokens``),
#: ``lm_head`` on the root. This matches what upstream auto-detection finds;
#: we pass it explicitly so a silent upstream fallback can never pick a
#: different site.
GEMMA4_LAYOUT = Layout("model.language_model")


def resolve_revision(
    repo_id: str, revision: str | None = None, *, token: str | None = None
) -> str:
    """Resolve ``revision`` (default ``main``) of ``repo_id`` to an immutable
    40-hex commit SHA via the HuggingFace Hub API.

    Raises:
        RuntimeError: If the repo is inaccessible or no commit SHA comes back.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    ref = revision or "main"
    try:
        info = HfApi(token=token).model_info(repo_id, revision=ref)
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"cannot resolve {repo_id}@{ref}: {exc}. Check network access and "
            f"(if the repo is gated) HF_TOKEN."
        ) from exc
    sha = getattr(info, "sha", None)
    if not (isinstance(sha, str) and len(sha) == 40):
        raise RuntimeError(
            f"{repo_id}@{ref} resolved without an immutable commit SHA (got {sha!r})"
        )
    if revision is not None and len(revision) == 40 and revision != sha:
        raise RuntimeError(
            f"pinned revision {revision} of {repo_id} resolved to a different "
            f"commit {sha}; refusing to continue"
        )
    return sha


class Gemma4LensModel(HFLensModel):
    """Upstream :class:`~jlens.hf.HFLensModel` with explicit BOS control and a
    pre-softcap readout path.

    BOS: upstream's ``force_bos=True`` sets ``tokenizer.add_bos_token``, which
    is inert on some fast tokenizers. We disable that mechanism and instead
    prepend ``bos_token_id`` in :meth:`encode` whenever the tokenizer did not,
    so the behaviour is deterministic and observable
    (:attr:`bos_prepended_by_tokenizer` records what the tokenizer itself did).
    """

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer: Any,
        *,
        layout: Layout | None = None,
        compile: bool = False,
        ensure_bos: bool = True,
    ) -> None:
        super().__init__(
            hf_model,
            tokenizer,
            layout=GEMMA4_LAYOUT if layout is None else layout,
            compile=compile,
            force_bos=False,
        )
        self.ensure_bos = ensure_bos
        #: Whether the tokenizer emitted BOS on its own for a probe string
        #: (``None`` if the tokenizer has no ``bos_token_id``).
        self.bos_prepended_by_tokenizer: bool | None = None
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is not None:
            probe = tokenizer("probe", return_tensors="pt").input_ids
            self.bos_prepended_by_tokenizer = bool(
                probe.shape[1] > 0 and int(probe[0, 0]) == int(bos)
            )

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        input_ids = super().encode(text, max_length=max_length)
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if (
            self.ensure_bos
            and bos is not None
            and (input_ids.shape[1] == 0 or int(input_ids[0, 0]) != int(bos))
        ):
            bos_column = torch.full(
                (input_ids.shape[0], 1),
                int(bos),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            input_ids = torch.cat([bos_column, input_ids], dim=1)[:, :max_length]
        return input_ids

    def unembed_presoftcap(self, residual: torch.Tensor) -> torch.Tensor:
        """Final norm + unembedding *without* Gemma's final-logit softcap —
        the paper's ``W_U norm(J h)`` readout."""
        target_device = self._lm_head.weight.device
        target_dtype = self._lm_head.weight.dtype
        return self._lm_head(
            self._final_norm(residual.to(target_dtype).to(target_device))
        )

    def unembed_pair(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(pre_softcap, softcapped)`` logits for ``residual``.

        ``softcapped`` equals what :meth:`~jlens.hf.HFLensModel.unembed`
        (Gemma's actual output pathway) produces; when the model has no
        softcap the two are identical.
        """
        pre = self.unembed_presoftcap(residual)
        if self._logit_softcap is None:
            return pre, pre
        return pre, self._logit_softcap * torch.tanh(pre / self._logit_softcap)


@dataclass
class ArchitectureReport:
    """Serialisable record of what :func:`verify_architecture` observed."""

    model_class: str = ""
    text_model_type: str = ""
    layout_path: str = ""
    n_layers: int = 0
    d_model: int = 0
    vocab_size: int = 0
    dense: bool = True
    tied_unembedding: bool = False
    final_logit_softcapping: float | None = None
    hidden_size_per_layer_input: int | None = None
    num_kv_shared_layers: int | None = None
    sliding_window: int | None = None
    n_full_attention_layers: int | None = None
    layer_scalars: list[float] = field(default_factory=list)
    layer_scalars_all_unit: bool = True
    params_frozen: bool = False
    dtype: str = ""
    device: str = ""
    bos_token_id: int | None = None
    bos_prepended_by_tokenizer: bool | None = None
    encode_starts_with_bos: bool | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def verify_architecture(
    model: Gemma4LensModel,
    *,
    expect_n_layers: int | None = None,
    expect_d_model: int | None = None,
    expect_vocab_size: int | None = None,
) -> ArchitectureReport:
    """Verify the assumptions the lens relies on; return a report.

    Hard failures (``ValueError``): MoE routing enabled, expected
    depth/width/vocab mismatch, any trainable parameter. Everything else —
    ``layer_scalar`` deviating from 1.0, tokenizer BOS behaviour, untied
    unembedding — is recorded in the report (with a warning) but does not
    raise, per the experiment protocol.
    """
    report = ArchitectureReport()
    hf_model = model._hf_model
    text_config = hf_model.config.get_text_config()

    report.model_class = type(hf_model).__name__
    report.text_model_type = getattr(text_config, "model_type", "")
    report.layout_path = model.layout.path
    report.n_layers = model.n_layers
    report.d_model = model.d_model
    report.vocab_size = int(model._lm_head.weight.shape[0])
    report.final_logit_softcapping = model._logit_softcap
    report.hidden_size_per_layer_input = getattr(
        text_config, "hidden_size_per_layer_input", None
    )
    report.num_kv_shared_layers = getattr(text_config, "num_kv_shared_layers", None)
    report.sliding_window = getattr(text_config, "sliding_window", None)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        report.n_full_attention_layers = sum(
            1 for t in layer_types if t == "full_attention"
        )

    if getattr(text_config, "enable_moe_block", False):
        raise ValueError(
            "enable_moe_block=True: MoE routing is out of scope for this "
            "adaptation; the residual-stream Jacobian estimator has only been "
            "validated on the dense configuration"
        )
    report.dense = True

    for name, expected, actual in (
        ("num_hidden_layers", expect_n_layers, report.n_layers),
        ("hidden_size", expect_d_model, report.d_model),
        ("vocab_size", expect_vocab_size, report.vocab_size),
    ):
        if expected is not None and actual != expected:
            raise ValueError(f"config expects {name}={expected}, model has {actual}")

    trainable = [n for n, p in hf_model.named_parameters() if p.requires_grad]
    if trainable:
        raise ValueError(
            f"{len(trainable)} parameters still require grad "
            f"(first: {trainable[0]}); the model must be frozen"
        )
    report.params_frozen = True

    report.tied_unembedding = model._lm_head.weight is model._embed_tokens.weight
    if not report.tied_unembedding:
        report.warnings.append(
            "lm_head.weight is not the same tensor as embed_tokens.weight "
            "(expected tied for Gemma 4)"
        )

    scalars: list[float] = []
    for block in model.layers:
        scalar = getattr(block, "layer_scalar", None)
        scalars.append(float(scalar.float().mean()) if scalar is not None else 1.0)
    report.layer_scalars = scalars
    report.layer_scalars_all_unit = all(abs(s - 1.0) < 1e-6 for s in scalars)
    if not report.layer_scalars_all_unit:
        report.warnings.append(
            f"layer_scalar deviates from 1.0 (min={min(scalars):.6g}, "
            f"max={max(scalars):.6g}); recorded, not fatal — the scalar is part "
            f"of the block output the lens reads"
        )

    report.dtype = str(model._embed_tokens.weight.dtype)
    report.device = str(model._embed_tokens.weight.device)

    bos = getattr(model.tokenizer, "bos_token_id", None)
    report.bos_token_id = int(bos) if bos is not None else None
    report.bos_prepended_by_tokenizer = model.bos_prepended_by_tokenizer
    if bos is not None:
        probe_ids = model.encode("architecture probe")
        report.encode_starts_with_bos = bool(int(probe_ids[0, 0]) == int(bos))
        if not report.encode_starts_with_bos:
            report.warnings.append(
                "encode() does not start sequences with BOS "
                "(ensure_bos disabled?); recorded, not fatal"
            )
    else:
        report.warnings.append("tokenizer has no bos_token_id")

    for message in report.warnings:
        logger.warning("verify_architecture: %s", message)
    return report


def load_gemma4(
    repo_id: str = GEMMA4_E4B_REPO,
    *,
    revision: str | None = None,
    tokenizer_repo_id: str | None = None,
    tokenizer_revision: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict | None = None,
    allow_model_load: bool = False,
    token: str | None = None,
) -> tuple[Gemma4LensModel, dict]:
    """Load ``google/gemma-4-E4B-it`` (or a compatible Gemma 4 checkpoint) at a
    pinned immutable revision and wrap it as a :class:`Gemma4LensModel`.

    The **unmodified multimodal model** is loaded — vision/audio towers stay in
    place (text-only execution never calls them). Any tower-removal
    optimisation must first demonstrate bit-identical text logits; none is
    performed here.

    Args:
        repo_id: HuggingFace model repo.
        revision: Ref to resolve (default ``main``). Resolved to an immutable
            SHA before any download; the SHA is used for every subsequent load.
        tokenizer_repo_id: Defaults to ``repo_id``.
        tokenizer_revision: Defaults to the resolved model SHA when the
            tokenizer repo is the model repo, else resolved from ``main``.
        dtype: Model dtype (checkpoint ships bf16).
        device_map: Passed to ``from_pretrained`` (e.g. ``"cuda"``; ``None``
            loads to CPU).
        allow_model_load: Must be ``True``; the ~16 GB download/load never
            happens implicitly.
        token: Optional HF token.

    Returns:
        ``(model, load_info)`` where ``load_info`` records repo ids, resolved
        revisions, dtype, device_map, and library versions.
    """
    if not allow_model_load:
        raise RuntimeError(
            "refusing to load Gemma 4 (~16 GB): pass allow_model_load=True "
            "(config key model.allow_model_load) to proceed"
        )
    import transformers

    model_sha = resolve_revision(repo_id, revision, token=token)
    tok_repo = tokenizer_repo_id or repo_id
    if tok_repo == repo_id and tokenizer_revision is None:
        tok_sha = model_sha
    else:
        tok_sha = resolve_revision(tok_repo, tokenizer_revision, token=token)
    logger.info("loading %s@%s (tokenizer %s@%s)", repo_id, model_sha, tok_repo, tok_sha)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tok_repo, revision=tok_sha, token=token
    )
    # AutoModelForCausalLM does not route Gemma4ForConditionalGeneration;
    # use the explicit class (assumption checked at import time).
    from transformers import Gemma4ForConditionalGeneration

    kwargs: dict[str, Any] = {"dtype": dtype, "revision": model_sha, "token": token}
    if device_map is not None:
        kwargs["device_map"] = device_map
    hf_model = Gemma4ForConditionalGeneration.from_pretrained(repo_id, **kwargs)

    model = Gemma4LensModel(hf_model, tokenizer)
    load_info = {
        "model_repo_id": repo_id,
        "model_revision": model_sha,
        "tokenizer_repo_id": tok_repo,
        "tokenizer_revision": tok_sha,
        "tokenizer_class": type(tokenizer).__name__,
        "dtype": str(dtype),
        "device_map": str(device_map),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    return model, load_info


@contextmanager
def softcap_disabled(model: HFLensModel):
    """Temporarily disable the final-logit softcap so ``unembed`` (and
    therefore ``JacobianLens.apply``) returns pre-softcap logits."""
    saved = model._logit_softcap
    model._logit_softcap = None
    try:
        yield
    finally:
        model._logit_softcap = saved


def apply_dual(
    lens: JacobianLens,
    model: HFLensModel,
    prompt: str,
    *,
    layers: list[int] | None = None,
    positions: list[int] | None = None,
    max_seq_len: int = 512,
    use_jacobian: bool = True,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, torch.Tensor], torch.Tensor]:
    """:meth:`jlens.lens.JacobianLens.apply` with **both** readout paths.

    Runs a single forward pass with the softcap disabled (pre-softcap logits,
    the paper's ``W_U norm(J h)`` convention), then derives the softcapped
    logits analytically — ``cap * tanh(x / cap)`` is exactly what
    :meth:`~jlens.hf.HFLensModel.unembed` would have produced. The cap is
    strictly monotonic, so rankings agree between the two paths; logit values
    and softmax probabilities differ.

    Returns:
        ``(lens_logits, model_logits, input_ids)`` where ``lens_logits[layer]``
        and ``model_logits`` are ``{"pre": ..., "capped": ...}`` dicts.
    """
    with softcap_disabled(model):
        lens_pre, model_pre, input_ids = lens.apply(
            model,
            prompt,
            layers=layers,
            positions=positions,
            max_seq_len=max_seq_len,
            use_jacobian=use_jacobian,
        )
    cap = model._logit_softcap

    def capped(logits: torch.Tensor) -> torch.Tensor:
        return cap * torch.tanh(logits / cap) if cap is not None else logits

    lens_logits = {
        layer: {"pre": pre, "capped": capped(pre)} for layer, pre in lens_pre.items()
    }
    model_logits = {"pre": model_pre, "capped": capped(model_pre)}
    return lens_logits, model_logits, input_ids


def probe_fit_cost(
    model: Gemma4LensModel | HFLensModel,
    prompt: str,
    source_layers: list[int],
    *,
    dim_batch: int = 4,
    max_seq_len: int = 48,
) -> dict:
    """Run :func:`jlens.fitting.jacobian_for_prompt` once at a small
    ``dim_batch`` and report wall time and (on CUDA) peak memory, so
    ``dim_batch`` can be scaled up from measurements rather than guesses.
    """
    import time

    from jlens.fitting import jacobian_for_prompt

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    jacobians, seq_len, n_valid = jacobian_for_prompt(
        model, prompt, source_layers, dim_batch=dim_batch, max_seq_len=max_seq_len
    )
    if cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        "dim_batch": dim_batch,
        "max_seq_len": max_seq_len,
        "seq_len": seq_len,
        "n_valid_positions": n_valid,
        "source_layers": list(source_layers),
        "wall_seconds": round(elapsed, 2),
        "jacobian_shapes": {l: list(j.shape) for l, j in jacobians.items()},
        "all_finite": all(bool(torch.isfinite(j).all()) for j in jacobians.values()),
        "peak_cuda_memory_gb": (
            round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None
        ),
    }
    del jacobians
    return result
