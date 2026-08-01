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

A capped control pool cannot produce a PASS
-------------------------------------------

The pool-matched control used to be capped at 16384 candidate atoms while the
lens searched its full dictionary — roughly 262k atoms for Gemma. That is not a
matched comparison. The greedy maximum correlation with a target grows with the
number of candidates searched, so a control restricted to a sixteenth of the
lens's pool *understates* what random directions can do, in the direction that
flatters the lens. Disclosing the bias in the record was not enough: the summary
still read that comparison as evidence and could turn it into a scientific PASS.

So the default is now an **equal-opportunity** comparison: the control pool has
exactly as many candidate atoms as the lens dictionary
(``max_control_pool_atoms=None``). Atoms are generated in chunks directly in the
lens's dtype and device, so peak memory is one control dictionary the same size
as the lens — on an L4 with Gemma E4B resident that is affordable, and each draw
is freed before the next is built.

Where an equal pool genuinely is not affordable, ``max_control_pool_atoms`` may
still be set. When the cap binds, the record's ``criterion_status`` becomes
``not_evaluated_pool_mismatch`` and
:func:`summarize_reconstruction_controls` refuses to mark the layer above
random. The answer is then *unknown*, which is different both from a pass and
from a failure, and it is reported as such rather than resolved in the lens's
favour. Setting ``require_pool_match=False`` is the explicit, fingerprinted way
to accept a mismatched comparison, and it downgrades the result to
``conditional_pool_mismatch`` — never a clean PASS.

There is still no absolute explained-variance threshold anywhere. The published
J-space result puts the top-k component at a median around 6-7% of a concept
vector's variance, so an absolute bar would reject a working lens.

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

#: How many random atoms to materialise at a time. Bounds peak memory while a
#: pool the size of a full Gemma dictionary is being built.
CONTROL_CHUNK_ATOMS = 16384

#: The criterion's verdict on one record.
STATUS_EVALUATED = "evaluated"
STATUS_NOT_EVALUATED = "not_evaluated_pool_mismatch"
STATUS_CONDITIONAL = "conditional_pool_mismatch"


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
            count. ``None`` — the default — means "exactly as many candidates
            as the lens dictionary", which is the only matched comparison.
            Setting a cap below the lens pool makes the comparison
            search-space-mismatched in the lens's favour.
        require_pool_match: When True (the default) a record whose control pool
            is smaller than the lens pool is marked
            :data:`STATUS_NOT_EVALUATED` and cannot contribute to a PASS.
            Setting it False is the explicit way to accept a mismatched
            comparison; the result is then labelled
            :data:`STATUS_CONDITIONAL` and still never reads as a clean pass.
        pool_ladder: Optional additional control-pool sizes to run alongside
            the primary one. Running the same comparison at increasing pool
            sizes shows whether the lens's margin is stable as the control gets
            more search freedom, which is the residual evidence available when
            an equal pool is genuinely unaffordable.
    """

    n_draws: int = 5
    seed: int = 20260731
    quantile: float = 0.95
    k_schedule: tuple[int, ...] = DEFAULT_K_SCHEDULE
    max_samples_per_layer: int = 8
    min_median_excess: float = 0.0
    max_control_pool_atoms: int | None = None
    require_pool_match: bool = True
    pool_ladder: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["k_schedule"] = list(self.k_schedule)
        payload["pool_ladder"] = list(self.pool_ladder)
        payload["schema_version"] = SCHEMA_VERSION
        payload["criterion_control"] = "pool_matched"
        payload["control_matching"] = [
            "candidate_pool_size (equal to the lens dictionary by default)",
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
        payload["capped_pool_cannot_pass"] = (
            "a control searching fewer candidates than the lens understates "
            "random performance, because the greedy maximum correlation grows "
            "with the number of candidates searched. Such a comparison is "
            "marked NOT EVALUATED and cannot produce a scientific PASS on its "
            "own"
        )
        payload["interpretation"] = (
            "the criterion is the pool-matched control, which gives random "
            "directions the same candidate pool, the same k, the same atom "
            "norms and the same pursuit as the lens, so the only difference "
            "left is which directions the atoms point in. If the pool cannot be "
            "matched, the criterion reports NOT EVALUATED rather than a result"
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

    Atoms are generated in chunks of :data:`CONTROL_CHUNK_ATOMS` and converted
    to ``dtype`` as they go, so a pool the size of a full Gemma dictionary never
    needs a transient float32 copy of itself. That is what makes the equal-pool
    default affordable next to a resident model.
    """
    if not len(reference_norms):
        raise ControlConfigurationError("a matched control needs at least one atom")
    count = int(n_atoms or len(reference_norms))
    if count < len(reference_norms):
        raise ControlConfigurationError(
            f"a pool of {count} cannot carry a support of {len(reference_norms)}"
        )
    generator = _generator(layer, *seed_parts, count)
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

    atoms = torch.empty(count, d_model, device=device, dtype=dtype)
    tiny = torch.finfo(torch.float32).tiny
    for start in range(0, count, CONTROL_CHUNK_ATOMS):
        stop = min(start + CONTROL_CHUNK_ATOMS, count)
        raw = torch.randn(stop - start, d_model, generator=generator, dtype=torch.float32)
        raw /= raw.norm(dim=-1, keepdim=True).clamp_min(tiny)
        raw *= scale[start:stop].unsqueeze(-1)
        atoms[start:stop] = raw.to(device=device, dtype=dtype)
        del raw
    return JSpaceDictionary(
        atoms,
        layer=layer,
        provenance={
            "kind": "matched_random_control",
            "n_atoms": count,
            "n_reference_norms": len(reference_norms),
            "norm_matched_to": matching,
            "d_model": d_model,
            "dtype": str(dtype),
            "device": str(device),
            "generated_in_chunks_of": CONTROL_CHUNK_ATOMS,
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
    # None means "match the lens exactly" — the only comparison in which the
    # control enjoys the same selection freedom the lens does.
    requested_pool = (
        dictionary.n_atoms
        if config.max_control_pool_atoms is None
        else min(dictionary.n_atoms, int(config.max_control_pool_atoms))
    )
    control_pool = max(requested_pool, len(support))

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

    # The gate. Same candidate pool, same k: the only thing left that differs
    # from the lens is which directions the atoms point in.
    pool_matched = run_control("pool_matched", control_pool, settings.k)
    # Reported for scale, never gated on — a k-atom control is beaten by any
    # dictionary with a larger selection pool, noise included.
    support_matched = run_control("support_matched", max(1, len(support)), len(support))

    # An optional ladder of smaller pools. It cannot rescue a mismatched
    # comparison, but it shows whether the lens's margin shrinks as the control
    # is given more search freedom — the residual evidence available when an
    # equal pool is unaffordable.
    ladder = [
        {
            "n_control_atoms": size,
            **{
                key: run_control("pool_ladder", size, settings.k)[key]
                for key in (
                    "median_explained_fraction",
                    "upper_bound_explained_fraction",
                )
            },
        }
        for size in sorted({
            max(len(support), min(int(rung), dictionary.n_atoms))
            for rung in config.pool_ladder
        })
    ]
    for rung in ladder:
        rung["jlens_excess"] = jlens_explained - rung["median_explained_fraction"]
    ladder_stable = bool(ladder) and all(rung["jlens_excess"] > 0.0 for rung in ladder)

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
    if not pool_capped:
        criterion_status = STATUS_EVALUATED
        status_reason = (
            "the control searched exactly as many candidate atoms as the lens"
        )
    elif config.require_pool_match:
        criterion_status = STATUS_NOT_EVALUATED
        status_reason = (
            f"the control searched {control_pool} candidate atoms against the "
            f"lens's {dictionary.n_atoms}. A smaller search pool understates "
            "random performance, so this comparison favours the lens and "
            "cannot establish that the lens beats random directions. Set "
            "max_control_pool_atoms=None to match the pool."
        )
    else:
        criterion_status = STATUS_CONDITIONAL
        status_reason = (
            f"require_pool_match was explicitly disabled with a control pool of "
            f"{control_pool} against the lens's {dictionary.n_atoms}; the "
            "comparison is search-space-mismatched in the lens's favour and is "
            "reported as conditional, never as a clean pass."
        )
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
        "pool_matched_exactly": not pool_capped,
        # The criterion's verdict on THIS record. A capped pool never says
        # "evaluated", so it can never be counted as evidence for the lens.
        "criterion_status": criterion_status,
        "criterion_status_reason": status_reason,
        "pool_ladder": ladder,
        "pool_ladder_stable": ladder_stable,
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
    not_evaluated: list[int] = []
    for layer in sorted(by_layer):
        rows = by_layer[layer]
        absolute = [float(r["jlens_explained_fraction"]) for r in rows]
        excess = [float(r["excess_explained_fraction"]) for r in rows]
        over_bound = [float(r["excess_over_random_bound"]) for r in rows]
        occupancies = [int(r["occupancy"]["estimated_occupancy"]) for r in rows]
        healthy = all(r["finite"] and r["nondegenerate"] for r in rows)
        median_excess = _median(excess)
        median_over_bound = _median(over_bound)
        # A layer is only evaluated when every one of its records compared the
        # lens against a control that searched the same pool. Records default
        # to "evaluated" so summaries of older artifacts still read, but a
        # capped pool is explicitly recorded and is refused here.
        statuses = {str(r.get("criterion_status", STATUS_EVALUATED)) for r in rows}
        layer_status = (
            STATUS_EVALUATED
            if statuses == {STATUS_EVALUATED}
            else (
                STATUS_CONDITIONAL
                if statuses <= {STATUS_EVALUATED, STATUS_CONDITIONAL}
                else STATUS_NOT_EVALUATED
            )
        )
        layer_above = bool(
            healthy
            and layer_status == STATUS_EVALUATED
            and median_excess > config.min_median_excess
            and median_over_bound > 0.0
        )
        if layer_above:
            above.append(layer)
        elif layer_status != STATUS_EVALUATED:
            not_evaluated.append(layer)
        layers[str(layer)] = {
            "layer": layer,
            "n_samples": len(rows),
            "criterion_status": layer_status,
            "criterion_status_reason": next(
                (
                    str(r.get("criterion_status_reason", ""))
                    for r in rows
                    if str(r.get("criterion_status", STATUS_EVALUATED)) != STATUS_EVALUATED
                ),
                "",
            ),
            "pool_matched_exactly": all(
                r.get("pool_matched_exactly", not r.get("control_pool_capped", False))
                for r in rows
            ),
            "pool_ladder_stable": all(r.get("pool_ladder_stable", False) for r in rows),
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

    evaluable = [
        entry for entry in layers.values() if entry["criterion_status"] == STATUS_EVALUATED
    ]
    return {
        "schema": SCHEMA_VERSION,
        "n_records": len(records),
        "config": config.to_dict(),
        "control_config_hash": config.fingerprint,
        "primary_layer": primary_layer,
        "by_layer": layers,
        "layers_above_random": above,
        "layers_not_evaluated": not_evaluated,
        # The report reads this: with no evaluable layer the criterion is
        # unknown, which is neither a pass nor a failure of the lens.
        "criterion_evaluable": bool(evaluable),
        "criterion_not_evaluated_reason": (
            ""
            if evaluable
            else next(
                (
                    entry["criterion_status_reason"]
                    for entry in layers.values()
                    if entry["criterion_status_reason"]
                ),
                "",
            )
        ),
        "all_pools_matched_exactly": all(
            entry["pool_matched_exactly"] for entry in layers.values()
        )
        if layers
        else False,
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
            "a larger selection pool. A control that searched FEWER candidates "
            "than the lens is not a matched comparison either: it understates "
            "random performance, so such a layer reports NOT EVALUATED and "
            "cannot contribute to a pass."
        ),
    }


__all__ = [
    "CONTROL_CHUNK_ATOMS",
    "DEFAULT_K_SCHEDULE",
    "SCHEMA_VERSION",
    "STATUS_CONDITIONAL",
    "STATUS_EVALUATED",
    "STATUS_NOT_EVALUATED",
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
