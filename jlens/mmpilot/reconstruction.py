# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Is the J-lens reconstruction better than matched random directions?

The pilot originally gated on an absolute number: the frozen lens had to
explain at least 50% of a held-out activation's variance or its coordinates
were declared meaningless. That threshold was wrong, and it was wrong in a way
that would have thrown out a working lens.

Anthropic's J-space work reports that the top-k J-space component of a concept
vector carries a **median of roughly 6-7%** of its variance, and that at median
occupancy the variance explained in excess of a same-size random-direction
control never exceeds about 10%. The finding is not that J-space reconstructs
activations well. It is that this *small* component carries disproportionate
causal and reportable content. Under a 50% gate, the published result itself
would read as a failed lens.

So absolute explained fraction is kept here as a **descriptive statistic only**,
and the sanity criterion asks the question that can actually be wrong: does the
frozen lens explain more than random directions matched to it?

Two controls, and why the obvious one is not enough
---------------------------------------------------

For each held-out text activation and each evaluated layer, ``h`` is decomposed
against the frozen J-lens dictionary with the configured nonnegative gradient
pursuit, giving a support of at most ``k`` atoms. It is then compared against
``n_draws`` deterministic random dictionaries, in the same hidden dimension,
dtype and device, under the same pursuit settings, in two matchings:

``support_matched``
    Exactly ``len(support)`` random atoms, each norm-matched one-for-one to the
    J-lens atom it stands in for. This is the narrowest reading of "same-size
    random-direction control" — and on its own it is **not a usable control**.
    The J-lens chooses its ``k`` atoms greedily out of a pool of hundreds of
    thousands; the support-matched control is handed ``k`` directions drawn
    without reference to ``h``. *Any* dictionary with a pool larger than ``k``
    beats it, including a dictionary of pure noise. It is computed and reported
    because it is informative about scale, and it is explicitly not the gate.

``pool_matched``
    The same number of **candidate** atoms as the J-lens dictionary, with atom
    norms resampled from the norms the J-lens decomposition actually used, and
    the same ``k``. The random control now enjoys the same selection freedom as
    the lens, so the comparison isolates the only thing left that differs: the
    atom *directions*. This is what the sanity criterion reads, and it can
    fail — a dictionary of unaligned directions does fail it.

Where the pool matching is approximate
--------------------------------------

A full Gemma dictionary is ~262k atoms; materialising several dense random
copies of that per activation is not affordable next to the model, so the
control pool is capped at :attr:`ReconstructionControlConfig.max_control_pool_atoms`.
When the cap binds, the control searches a *smaller* pool than the lens does
and therefore understates what random directions could achieve. That bias
favours the lens, it is disclosed in every record as
``pool_selection_bias_factor`` (the ``sqrt(log N_lens / log N_control)`` scaling
of the greedy maximum correlation among isotropic directions), and it means a
PASS should be read as "clearly above noise", not as a precise effect size.

Occupancy
---------

:func:`occupancy` estimates the largest ``k`` at which the J-lens is still
buying more than the matched random control buys, by comparing *marginal*
reconstruction improvement along a short ``k`` schedule. This is inspired by
the paper's occupancy measure and is **not** a replication of it: the schedule
is short, the sample is a handful of held-out text activations at two layers,
and the control is support-matched as described above. It is reported as an
estimate with its procedure attached, never as the paper's number.

The ``k`` curve itself is exact rather than approximated. ``gradient_pursuit``
is greedy and records ``residual_norm_history``, so the residual after ``k``
iterations of a ``k_max`` run is bit-identical to a run configured with that
``k`` — provided no early stop fires, which is why this module refuses a
non-zero ``tol_relative_residual``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

import torch

from jlens.mmpilot.store import payload_checksum
from jlens.pursuit import JSpaceDictionary, PursuitSettings, gradient_pursuit

SCHEMA_VERSION = "jlens.mmpilot.reconstruction_control.v1"

#: The short k schedule the occupancy estimate walks. Deliberately small: this
#: is a pilot diagnostic, not a hyperparameter sweep.
DEFAULT_K_SCHEDULE = (1, 2, 4, 8, 16, 25)


class ControlConfigurationError(RuntimeError):
    """The control cannot be run faithfully under these settings."""


@dataclass(frozen=True)
class ReconstructionControlConfig:
    """How the matched-random comparison is drawn and read.

    Args:
        n_draws: Independent random dictionaries per activation. Kept small —
            the pilot needs a bound, not a tight one.
        seed: Base seed. Every draw's generator is seeded from this plus the
            layer and sample id, so a resumed run reproduces the same controls.
        quantile: Upper quantile of the random draws used as the bound the
            J-lens must beat.
        k_schedule: The occupancy schedule.
        max_samples_per_layer: Cap on held-out activations evaluated per layer.
        min_median_excess: The median excess over the random median must
            exceed this. Zero means "strictly better than random".
        max_control_pool_atoms: Cap on the pool-matched control's candidate
            count, so several dense random dictionaries fit next to the model.
            When it binds, the control is conservative and says so.
    """

    n_draws: int = 5
    seed: int = 20260731
    quantile: float = 0.95
    k_schedule: tuple[int, ...] = DEFAULT_K_SCHEDULE
    max_samples_per_layer: int = 8
    min_median_excess: float = 0.0
    max_control_pool_atoms: int = 16384

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["k_schedule"] = list(self.k_schedule)
        payload["schema_version"] = SCHEMA_VERSION
        payload["criterion_control"] = "pool_matched"
        payload["control_matching"] = [
            "candidate_pool_size (capped)",
            "sparsity k",
            "atom_l2_norms (resampled from the atoms the lens used)",
            "hidden_dimension",
            "dtype",
            "device",
            "pursuit_settings",
        ]
        payload["control_not_matched"] = ["atom_directions"]
        payload["support_matched_control_is_reported_not_gated"] = (
            "a control with only k candidate atoms is beaten by ANY dictionary "
            "with a larger selection pool, including pure noise, so it cannot "
            "gate anything; it is reported for scale"
        )
        payload["interpretation"] = (
            "the criterion is the pool-matched control, which gives random "
            "directions the same selection freedom as the lens. Where the pool "
            "cap binds, the control searches fewer candidates than the lens and "
            "so understates random performance — a bias favouring the lens, "
            "reported per record as pool_selection_bias_factor"
        )
        return payload

    @property
    def fingerprint(self) -> str:
        return payload_checksum(self.to_dict())


def default_config() -> ReconstructionControlConfig:
    return ReconstructionControlConfig()


# ------------------------------------------------------------- random controls


def _generator(*parts: object) -> torch.Generator:
    """A deterministic CPU generator seeded from ``parts``.

    Python's ``hash`` is salted per process; this is not, so a resumed run
    draws the same controls it drew the first time.
    """
    digest = payload_checksum(["reconstruction-control", *[str(part) for part in parts]])
    seed = int(digest.split(":")[1][:15], 16) % (2**63 - 1)
    return torch.Generator().manual_seed(seed)


def matched_random_dictionary(
    reference_norms: Sequence[float],
    d_model: int,
    *,
    layer: int,
    seed_parts: Sequence[object],
    n_atoms: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> JSpaceDictionary:
    """Random atoms carrying the J-lens's atom norms.

    With ``n_atoms`` unset the dictionary has exactly ``len(reference_norms)``
    atoms, each scaled to the norm of the J-lens atom it stands in for — the
    support-matched control. With ``n_atoms`` set larger, norms are resampled
    deterministically from ``reference_norms`` to fill the pool, so every
    control atom still carries a norm the J-lens itself used.

    Either way the control differs from the J-lens in *direction* only.
    """
    if not len(reference_norms):
        raise ControlConfigurationError("a matched control needs at least one atom")
    count = int(n_atoms or len(reference_norms))
    if count < len(reference_norms):
        raise ControlConfigurationError(
            f"a pool of {count} cannot carry a support of {len(reference_norms)}"
        )
    generator = _generator(layer, *seed_parts, count)
    raw = torch.randn(count, d_model, generator=generator, dtype=torch.float32)
    norms = raw.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    reference = torch.tensor([float(norm) for norm in reference_norms], dtype=torch.float32)
    if count == len(reference_norms):
        scale = reference
        matching = "selected j-lens atoms, one for one"
    else:
        picks = torch.randint(
            len(reference_norms), (count,), generator=generator, dtype=torch.long
        )
        scale = reference[picks]
        matching = "resampled from the norms of the atoms the j-lens selected"
    atoms = (raw / norms) * scale.unsqueeze(-1)
    return JSpaceDictionary(
        atoms.to(device=device, dtype=dtype),
        layer=layer,
        provenance={
            "kind": "matched_random_control",
            "n_atoms": count,
            "n_reference_norms": len(reference_norms),
            "norm_matched_to": matching,
            "d_model": d_model,
            "dtype": str(dtype),
            "device": str(device),
        },
    )


# ------------------------------------------------------------- the k curve


def explained_fraction_curve(
    result, item: int, k_schedule: Sequence[int]
) -> dict[int, float]:
    """``{k: explained fraction after k greedy atoms}`` for one batch item.

    Read off ``residual_norm_history``, which is exact for a greedy pursuit:
    the state after ``k`` iterations of a ``k_max`` run is the state a ``k``
    run would have reached.
    """
    history = result.residual_norm_history[item]
    target_norm = float(result.target_norm[item])
    if target_norm <= 0:
        return {int(k): 0.0 for k in k_schedule}
    out: dict[int, float] = {}
    for k in k_schedule:
        index = min(int(k), history.shape[0] - 1)
        residual = float(history[index])
        out[int(k)] = 1.0 - (residual * residual) / (target_norm * target_norm)
    return out


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered)) - 1)))
    return ordered[index]


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _marginals(curve: Mapping[int, float], k_schedule: Sequence[int]) -> dict[int, float]:
    out: dict[int, float] = {}
    previous = 0.0
    for k in k_schedule:
        out[int(k)] = curve[int(k)] - previous
        previous = curve[int(k)]
    return out


def occupancy(
    jlens_curve: Mapping[int, float],
    random_bound_curve: Mapping[int, float],
    k_schedule: Sequence[int],
) -> dict:
    """Largest ``k`` whose marginal J-lens gain still beats the random bound.

    Walks the schedule from the smallest ``k`` and stops at the first ``k``
    where the J-lens marginal improvement no longer exceeds the matched random
    control's bound on the same marginal — "remains above", read contiguously.
    Zero means the J-lens never beat the control, not even on the first atom.
    """
    jlens_marginal = _marginals(jlens_curve, k_schedule)
    random_marginal = _marginals(random_bound_curve, k_schedule)
    estimate = 0
    per_k = []
    still_above = True
    for k in k_schedule:
        above = jlens_marginal[int(k)] > random_marginal[int(k)]
        per_k.append(
            {
                "k": int(k),
                "jlens_marginal": jlens_marginal[int(k)],
                "random_bound_marginal": random_marginal[int(k)],
                "above": bool(above),
            }
        )
        if still_above and above:
            estimate = int(k)
        elif still_above:
            still_above = False
    return {
        "estimated_occupancy": estimate,
        "per_k": per_k,
        "definition": (
            "largest k in the schedule for which the J-lens marginal "
            "reconstruction improvement stayed above the matched random "
            "control's upper-quantile marginal, walking the schedule upward"
        ),
        "is_exact_replication_of_published_occupancy": False,
        "approximation": (
            "short k schedule, a handful of held-out text activations at the "
            "pilot's two layers, and a support-matched rather than "
            "pool-matched random control"
        ),
    }


# --------------------------------------------------------------- one activation


def reconstruction_control_record(
    activation: torch.Tensor,
    dictionary: JSpaceDictionary,
    settings: PursuitSettings,
    *,
    config: ReconstructionControlConfig,
    sample_id: str,
    layer: int,
    modality: str,
    split: str,
    activation_checksum: str,
    lens_checksum: str,
) -> dict:
    """J-lens reconstruction next to ``n_draws`` matched random controls.

    Raises:
        ControlConfigurationError: If the pursuit is configured to stop early
            on a residual tolerance, which would break the exactness of the
            ``k`` curve read off ``residual_norm_history``.
    """
    if settings.tol_relative_residual:
        raise ControlConfigurationError(
            "the k curve is read from residual_norm_history, which is only "
            "exact when no early stop fires; set tol_relative_residual=0.0"
        )
    schedule = tuple(k for k in config.k_schedule if k <= settings.k)
    if not schedule:
        raise ControlConfigurationError(
            f"k schedule {config.k_schedule} has nothing at or below k={settings.k}"
        )

    target = activation.to(device=dictionary.device, dtype=torch.float32).unsqueeze(0)
    jlens = gradient_pursuit(target, dictionary, settings)
    support = jlens.token_ids[0][: int(jlens.n_selected[0])].tolist()
    coefficients = jlens.coefficients[0][: int(jlens.n_selected[0])].tolist()
    jlens_curve = explained_fraction_curve(jlens, 0, schedule)
    jlens_explained = float(jlens.explained_fraction[0])

    reference_norms = [float(dictionary.atom_norms[int(atom)]) for atom in support]
    control_pool = min(dictionary.n_atoms, config.max_control_pool_atoms)
    control_pool = max(control_pool, len(support))

    def run_control(kind: str, n_atoms: int, k: int) -> dict:
        control_settings = PursuitSettings(
            k=max(1, min(k, n_atoms)),
            normalize_atoms=settings.normalize_atoms,
            refine_steps=settings.refine_steps,
            tol_relative_residual=0.0,
            max_backtracks=settings.max_backtracks,
            correlation_chunk_size=settings.correlation_chunk_size,
        )
        draws = []
        for draw in range(config.n_draws):
            control = matched_random_dictionary(
                reference_norms,
                dictionary.d_model,
                layer=layer,
                n_atoms=n_atoms,
                seed_parts=(config.seed, sample_id, kind, draw),
                device=dictionary.device,
                dtype=dictionary.atoms.dtype,
            )
            result = gradient_pursuit(target, control, control_settings)
            draws.append(
                {
                    "draw": draw,
                    "explained_fraction": float(result.explained_fraction[0]),
                    "curve": explained_fraction_curve(
                        result, 0, [k for k in schedule if k <= control_settings.k]
                    ),
                    "n_selected": int(result.n_selected[0]),
                }
            )
            del control
        values = [draw["explained_fraction"] for draw in draws]
        return {
            "kind": kind,
            "n_control_atoms": n_atoms,
            "k": control_settings.k,
            "explained_fractions": values,
            "mean_explained_fraction": sum(values) / len(values) if values else 0.0,
            "median_explained_fraction": _median(values),
            "stdev_explained_fraction": _stdev(values),
            "upper_bound_explained_fraction": _quantile(values, config.quantile),
            "bound_quantile": config.quantile,
            "bound_curve": {
                int(k): _quantile(
                    [draw["curve"].get(int(k), 0.0) for draw in draws], config.quantile
                )
                for k in schedule
            },
            "draws": draws,
        }

    # The gate. Same candidate pool (capped), same k: the only thing left that
    # differs from the lens is which directions the atoms point in.
    pool_matched = run_control("pool_matched", control_pool, settings.k)
    # Reported for scale, never gated on — a k-atom control is beaten by any
    # dictionary with a larger selection pool, noise included.
    support_matched = run_control("support_matched", max(1, len(support)), len(support))

    random_values = pool_matched["explained_fractions"]
    random_median = pool_matched["median_explained_fraction"]
    random_bound = pool_matched["upper_bound_explained_fraction"]
    random_bound_curve = pool_matched["bound_curve"]
    finite = (
        math.isfinite(jlens_explained)
        and all(math.isfinite(value) for value in random_values)
        and all(math.isfinite(float(c)) for c in coefficients)
    )
    n_active = sum(1 for c in coefficients if c > 0)
    pool_capped = control_pool < dictionary.n_atoms
    return {
        "schema": SCHEMA_VERSION,
        "sample_id": sample_id,
        "layer": int(layer),
        "modality": modality,
        "split": split,
        "k": int(settings.k),
        "n_support_atoms": len(support),
        "n_active": n_active,
        "target_norm": float(jlens.target_norm[0]),
        # Descriptive only. A small number here is expected for a sparse
        # workspace and is never on its own a reason to call the lens broken.
        "jlens_explained_fraction": jlens_explained,
        "jlens_curve": jlens_curve,
        # The criterion reads the pool-matched control; these mirror it.
        "criterion_control": "pool_matched",
        "random_explained_fractions": random_values,
        "random_mean_explained_fraction": pool_matched["mean_explained_fraction"],
        "random_median_explained_fraction": random_median,
        "random_stdev_explained_fraction": pool_matched["stdev_explained_fraction"],
        "random_upper_bound_explained_fraction": random_bound,
        "random_bound_quantile": config.quantile,
        "random_bound_curve": random_bound_curve,
        "excess_explained_fraction": jlens_explained - random_median,
        "excess_over_random_bound": jlens_explained - random_bound,
        "above_random_bound": jlens_explained > random_bound,
        "controls": {
            "pool_matched": pool_matched,
            "support_matched": support_matched,
        },
        "lens_pool_size": int(dictionary.n_atoms),
        "control_pool_size": int(control_pool),
        "control_pool_capped": bool(pool_capped),
        "pool_selection_bias_factor": (
            math.sqrt(math.log(dictionary.n_atoms) / math.log(control_pool))
            if pool_capped and control_pool > 1
            else 1.0
        ),
        "pool_bias_direction": (
            "control searches a smaller pool than the lens, so it understates "
            "random performance and the comparison favours the lens"
            if pool_capped
            else "control and lens search pools of the same size"
        ),
        "occupancy": occupancy(jlens_curve, random_bound_curve, schedule),
        "n_draws": config.n_draws,
        "seed": config.seed,
        "control_config_hash": config.fingerprint,
        "pursuit_settings": asdict(settings),
        "activation_checksum": activation_checksum,
        "lens_checksum": lens_checksum,
        "finite": bool(finite),
        "nondegenerate": bool(
            finite and n_active > 0 and float(jlens.target_norm[0]) > 0
        ),
        "convergence_status": jlens.stop_reasons[0],
    }


# ----------------------------------------------------------------- the summary


@dataclass
class ReconstructionControlSummary:
    """Per-layer roll-up, and the evidence the sanity criterion reads."""

    records: list[dict] = field(default_factory=list)
    config: ReconstructionControlConfig = field(default_factory=default_config)
    primary_layer: int | None = None

    def to_dict(self) -> dict:
        return summarize_reconstruction_controls(
            self.records, config=self.config, primary_layer=self.primary_layer
        )


def summarize_reconstruction_controls(
    records: Sequence[Mapping],
    *,
    config: ReconstructionControlConfig | None = None,
    primary_layer: int | None = None,
) -> dict:
    """Roll ``records`` up per layer. Empty input yields ``n_records == 0``."""
    config = config or default_config()
    by_layer: dict[int, list[Mapping]] = {}
    for record in records:
        by_layer.setdefault(int(record["layer"]), []).append(record)

    layers: dict[str, dict] = {}
    above: list[int] = []
    for layer in sorted(by_layer):
        rows = by_layer[layer]
        absolute = [float(r["jlens_explained_fraction"]) for r in rows]
        excess = [float(r["excess_explained_fraction"]) for r in rows]
        over_bound = [float(r["excess_over_random_bound"]) for r in rows]
        occupancies = [int(r["occupancy"]["estimated_occupancy"]) for r in rows]
        healthy = all(r["finite"] and r["nondegenerate"] for r in rows)
        median_excess = _median(excess)
        median_over_bound = _median(over_bound)
        layer_above = bool(
            healthy
            and median_excess > config.min_median_excess
            and median_over_bound > 0.0
        )
        if layer_above:
            above.append(layer)
        layers[str(layer)] = {
            "layer": layer,
            "n_samples": len(rows),
            # Descriptive.
            "median_explained_fraction": _median(absolute),
            "mean_explained_fraction": sum(absolute) / len(absolute) if absolute else 0.0,
            "min_explained_fraction": min(absolute) if absolute else 0.0,
            # The criterion reads these.
            "median_random_median_explained_fraction": _median(
                [float(r["random_median_explained_fraction"]) for r in rows]
            ),
            "median_random_upper_bound": _median(
                [float(r["random_upper_bound_explained_fraction"]) for r in rows]
            ),
            "median_excess_explained_fraction": median_excess,
            "median_excess_over_random_bound": median_over_bound,
            "fraction_samples_above_random_bound": (
                sum(1 for r in rows if r["above_random_bound"]) / len(rows)
                if rows
                else 0.0
            ),
            "above_random": layer_above,
            "median_estimated_occupancy": int(_median(occupancies)),
            "all_finite": all(r["finite"] for r in rows),
            "all_nondegenerate": all(r["nondegenerate"] for r in rows),
            # Reported for scale only; never part of ``above_random``.
            "median_support_matched_excess": _median(
                [
                    float(r["jlens_explained_fraction"])
                    - float(
                        r.get("controls", {})
                        .get("support_matched", {})
                        .get("median_explained_fraction", 0.0)
                    )
                    for r in rows
                    if r.get("controls")
                ]
            ),
            "control_pool_capped": any(r.get("control_pool_capped") for r in rows),
            "max_pool_selection_bias_factor": max(
                (float(r.get("pool_selection_bias_factor", 1.0)) for r in rows),
                default=1.0,
            ),
        }

    return {
        "schema": SCHEMA_VERSION,
        "n_records": len(records),
        "config": config.to_dict(),
        "control_config_hash": config.fingerprint,
        "primary_layer": primary_layer,
        "by_layer": layers,
        "layers_above_random": above,
        "evaluated_on": "held-out text activations only",
        "criterion_control": "pool_matched",
        "reading": (
            "Absolute explained fraction is descriptive. A sparse workspace is "
            "expected to explain only a small share of total activation "
            "variance — the published J-space result reports a median around "
            "6-7% — so the criterion is excess over matched random controls, "
            "not an absolute level. The gating control gives random directions "
            "the same candidate pool and the same k as the lens; the "
            "support-matched numbers are reported for scale and cannot gate "
            "anything, since a k-atom control is beaten by any dictionary with "
            "a larger selection pool."
        ),
    }


__all__ = [
    "DEFAULT_K_SCHEDULE",
    "SCHEMA_VERSION",
    "ControlConfigurationError",
    "ReconstructionControlConfig",
    "ReconstructionControlSummary",
    "default_config",
    "explained_fraction_curve",
    "matched_random_dictionary",
    "occupancy",
    "reconstruction_control_record",
    "summarize_reconstruction_controls",
]
