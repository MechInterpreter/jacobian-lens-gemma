# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The paper's *contiguous layer band* clamp: design, admissibility, verdicts.

Anthropic's primary swap protocol applies the exact two-coordinate operation

    c = pinv(V) h,   h_patched = h + alpha * V (sigma(c) - c)

at **every original prompt position**, at **every physical layer of a
contiguous workspace band**, recomputing the coordinates from each layer's own
activation and its own J-lens vectors. This module is the design half of that
protocol for Gemma 4 E4B: which layers may enter a band, which bands are
predeclared, what a band run binds itself to, and how its results are
aggregated and judged.

The intervention itself is **not** re-implemented here. It is
:func:`jlens.mmpilot.coordinate_swap.coordinate_swap_band`, which already
installs one hook per band layer, recomputes each layer's coordinates from that
layer's current activation, and never touches a teacher-forced candidate token.

What this replaces, and what it does not
========================================

The completed v2 study (:mod:`jlens.mmpilot.paper_reasoning_swap`) applied
**one** exchange at **one** independently confirmed layer per forward pass, on
the sampled grid 32/35/38/40, and recorded ``repeated_exchange_forbidden`` in
its own design record. That was a real experiment and its artifacts stand
unchanged. Its *justification* does not generalize:

    Exact exchange is an involution, so applying **one fixed update** twice
    cancels. A band clamp is not that. Each physical layer has its own
    ``W_U @ J_l``, so each layer's ``V`` is a different pair of directions, and
    each layer re-reads ``c`` from the activation that actually arrived there.
    Two consecutive band layers cancel only to the extent that their bases and
    their coordinates agree, which in a real transformer they do not.

So a contiguous band clamp is admissible, and this module builds one. What a
band still requires is that **every physical layer inside it** has an
independently confirmed, pinned lens — because an unconfirmed lens has no
established relationship to the model's computation, and a swap read through it
would exchange coordinates that mean nothing. The confirmed grid
``[32, 35, 38, 40]`` is **not** a band: 33, 34, 36, 37 and 39 are missing, and
:func:`assert_contiguous` refuses to call that set a range.

The two onset statistics, and why both are reported
===================================================

Over **suffix** bands ``[s .. end]`` the tested bands are nested: a later start
is a subset of an earlier one. That makes "the earliest start at which an arm
becomes effective" monotone-degenerate for any arm whose effect is carried by
the deepest layers — it is simply the earliest tested start. So every timing
result reports two numbers per arm:

``earliest_effective_start``
    The first tested band start that passes. Reported because it is the
    literal reading of "when does this become effective", and because a
    non-monotone measurement (cancellation across a long band) would show up
    here and nowhere else.

``deepest_effective_start``
    The last tested band start that still passes — the depth by which the
    representation has been consumed or committed. This is the informative
    statistic under nested suffix bands, and it is what
    :func:`band_onset_timing` classifies on: an intermediate concept that stops
    being causally editable *before* the answer does is evidence that it was
    computed first and then used.

Neither number is an exact physical-layer onset. Both are properties of the
predeclared band grid, and every payload says so.

Scoring
=======

The endpoint here is **restricted-candidate**: the fraction of trials in which
the downstream target answer outranks the other predeclared, externally scored
candidates. It is a forced-choice preference over a two-element set, measured by
teacher-forced conditional sequence likelihood, and it is *not* the endpoint
Anthropic's systematic evaluation reports. Theirs inspects the complete
next-token distribution with no answer appended and counts success only when the
target token is the **global** argmax over the whole vocabulary; that endpoint
lives in :mod:`jlens.mmpilot.full_vocabulary` and this module never computes it.

Rank, log-prob and margin are secondary and never promote a trial to a success —
a positive margin without candidate-set top-1 is reported as
:data:`PARTIAL_MOVEMENT`, which is a movement statistic, not a categorical one.
Identity replacement is scored as an **intervention-integrity diagnostic** and
can never, on its own, produce a reasoning GO.

Every field this module computes is therefore
:data:`~jlens.mmpilot.full_vocabulary.ENDPOINT_RESTRICTED_CANDIDATE`. See
``docs/endpoint_semantics.md`` for what each endpoint does and does not license.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from jlens.mmpilot.coordinate_swap import (
    PRIMARY_POSITION_RULE,
    LayerBand,
    build_layer_band,
)
from jlens.mmpilot.full_vocabulary import ENDPOINT_RESTRICTED_CANDIDATE
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "ALPHA_ROLES",
    "BAND_ARMS",
    "BAND_CONDITIONS",
    "BAND_DESIGN_VERSION",
    "BAND_INTERVENTION_FAMILY",
    "BAND_SWAP_VERSION",
    "BAND_TIMING_VERSION",
    "BAND_VERDICT_VERSION",
    "CONDITION_ALPHA",
    "PARTIAL_MOVEMENT",
    "PRIMARY_ALPHA",
    "SECONDARY_ALPHA",
    "BandDesignRefused",
    "BandLensRow",
    "BandSwapThresholds",
    "assert_contiguous",
    "band_design_record",
    "band_key",
    "band_onset_timing",
    "band_reasoning_verdict",
    "band_swap_fingerprint",
    "band_trial_record",
    "bootstrap_interval",
    "build_band",
    "contiguous_runs",
    "controls_for_condition",
    "format_lens_inventory",
    "largest_admissible_band",
    "lens_inventory",
    "predeclare_suffix_bands",
    "read_band_units",
    "summarize_band_cells",
]

#: Bound into every band artifact. A change refuses to resume older results.
BAND_SWAP_VERSION = "mmpilot.anthropic_contiguous_band_coordinate_swap.v1"

#: The intervention family. Distinct from
#: :data:`jlens.mmpilot.coordinate_swap.INTERVENTION_FAMILY` (single layer or
#: arbitrary layer set) and from the v2 single-layer family, so no unit from
#: either can be read as a band unit.
BAND_INTERVENTION_FAMILY = "anthropic_contiguous_band_coordinate_swap"

BAND_DESIGN_VERSION = "mmpilot.anthropic_band_design.v1"
BAND_VERDICT_VERSION = "mmpilot.anthropic_band_reasoning_verdict.v1"
BAND_TIMING_VERSION = "mmpilot.anthropic_band_onset_timing.v1"

#: The paper's exact exchange. The primary condition, never swept per sample.
PRIMARY_ALPHA = 1.0

#: The paper's separately reported "double strength" condition. It is an
#: amplified extrapolation, not a stronger exchange, and every artifact says so.
SECONDARY_ALPHA = 2.0

ALPHA_ROLES: dict[str, str] = {
    "1.0": "primary_exact_coordinate_exchange",
    "2.0": "secondary_fixed_double_strength_extrapolation",
}

#: The two uses of the same operator that the timing comparison contrasts.
BAND_ARMS = ("intermediate", "answer")

#: Every condition run at every band, for both arms. ``zero`` is alpha=0
#: through the identical code path and therefore serves both alphas.
BAND_CONDITIONS = (
    "swap_alpha1",
    "swap_alpha2",
    "zero",
    "random_alpha1",
    "random_alpha2",
    "unrelated_alpha1",
    "unrelated_alpha2",
)

CONDITION_ALPHA: dict[str, float] = {
    "swap_alpha1": 1.0,
    "swap_alpha2": 2.0,
    "zero": 0.0,
    "random_alpha1": 1.0,
    "random_alpha2": 2.0,
    "unrelated_alpha1": 1.0,
    "unrelated_alpha2": 2.0,
}

#: A trial whose target margin improved but whose top-1 did not change. Counted
#: and reported; never counted as a success.
PARTIAL_MOVEMENT = "partial_movement_not_top1"

#: The readout whose candidate-set top-1 is this study's causal endpoint, and
#: the readout that is only ever a diagnostic. Neither is a global argmax.
PRIMARY_READOUT = "property"
DIAGNOSTIC_READOUT = "identity"


class BandDesignRefused(RuntimeError):
    """A band, a lens inventory, or a scored cell cannot support the design."""


# --------------------------------------------------------------- contiguity


def contiguous_runs(layers: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Maximal contiguous ``(start, end)`` runs in ``layers``, ascending."""
    ordered = sorted({int(layer) for layer in layers})
    runs: list[tuple[int, int]] = []
    for layer in ordered:
        if runs and layer == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], layer)
        else:
            runs.append((layer, layer))
    return tuple(runs)


def assert_contiguous(layers: Sequence[int], *, what: str = "layer band") -> tuple[int, ...]:
    """Refuse a layer set with a gap, naming the missing physical layers.

    The confirmed sampled grid ``[32, 35, 38, 40]`` is the case this exists
    for: it is four independently confirmed layers, and describing it as the
    band 32-40 would claim that 33, 34, 36, 37 and 39 were patched when they
    were not.

    Raises:
        BandDesignRefused: If the set is empty, has a repeat, or has a gap.
    """
    ordered = [int(layer) for layer in layers]
    if not ordered:
        raise BandDesignRefused(f"{what} is empty")
    unique = sorted(set(ordered))
    if len(unique) != len(ordered):
        raise BandDesignRefused(f"{what} {ordered} repeats a layer")
    missing = [
        layer for layer in range(unique[0], unique[-1] + 1) if layer not in set(unique)
    ]
    if missing:
        raise BandDesignRefused(
            f"{what} {unique} is not contiguous: physical layer(s) {missing} are "
            f"inside the range {unique[0]}-{unique[-1]} and are not members. A "
            "sampled grid is not a band — calling it one would report layers as "
            "patched that no hook ever touched."
        )
    return tuple(unique)


def build_band(
    start: int,
    end: int,
    *,
    usable_layers: Sequence[int],
    n_layers: int | None = None,
) -> LayerBand:
    """One contiguous, confirmation-gated band — :func:`build_layer_band`.

    Thin by design: the gate that decides whether a layer may define exchanged
    coordinates already exists, and a second implementation of it would be a
    second thing to keep in step.
    """
    return build_layer_band(
        int(start), int(end), validated_layers=usable_layers, n_layers=n_layers
    )


def largest_admissible_band(
    usable_layers: Sequence[int], *, low: int, high: int
) -> tuple[int, int] | None:
    """The longest contiguous run of usable layers inside ``[low, high]``.

    The tie rule is fixed here, before any causal number exists: **longest
    wins; ties go to the shallowest start.** Shallowest rather than deepest
    because the scientific question is how early the intermediate concept can
    be edited, so a tie resolved toward depth would resolve it toward the
    answer the study is trying to test.

    Returns ``None`` when no layer in the window is usable.
    """
    window = {
        int(layer)
        for layer in usable_layers
        if int(low) <= int(layer) <= int(high)
    }
    runs = contiguous_runs(sorted(window))
    if not runs:
        return None
    return max(runs, key=lambda run: (run[1] - run[0] + 1, -run[0]))


def band_key(band: LayerBand | Sequence[int]) -> str:
    """``"32-40"`` — the stable string a stored unit is keyed and grouped by."""
    layers = tuple(band.layers) if isinstance(band, LayerBand) else tuple(
        int(layer) for layer in band
    )
    assert_contiguous(layers, what="band key")
    return f"{layers[0]}-{layers[-1]}"


def predeclare_suffix_bands(
    *,
    starts: Sequence[int],
    end: int,
    usable_layers: Sequence[int],
    n_layers: int | None = None,
) -> tuple[LayerBand, ...]:
    """The frozen suffix-band grid, every layer of every band confirmed.

    Args:
        starts: Band start layers, e.g. ``(32, 35, 38, 40)``. They are *starts*
            of contiguous ranges, not a sampled layer grid: ``32`` means the
            band 32,33,...,end, with every one of those layers patched.
        end: The common last layer of every band.

    Raises:
        BandDesignRefused: If ``starts`` is empty or unordered.
        LayerBandError: If any physical layer inside any band lacks a confirmed
            lens. Bands are never trimmed to fit the lenses on hand.
    """
    ordered = sorted({int(value) for value in starts})
    if not ordered:
        raise BandDesignRefused("no band starts were predeclared")
    if ordered[-1] > int(end):
        raise BandDesignRefused(
            f"band start {ordered[-1]} is deeper than the band end {end}"
        )
    return tuple(
        build_band(start, int(end), usable_layers=usable_layers, n_layers=n_layers)
        for start in ordered
    )


# ------------------------------------------------------------ lens inventory


@dataclass(frozen=True)
class BandLensRow:
    """One physical layer's lens status, read from artifacts rather than named.

    Every field is either copied from a discovered artifact or ``None``. A
    ``None`` is a statement that the artifact did not record it, never a
    default, and a row with any required field missing is not usable.
    """

    layer: int
    lens_path: str | None = None
    lens_checksum: str | None = None
    fit_scale: int | None = None
    fit_corpus: str | None = None
    validation_set: str | None = None
    confirmation_verdict: str | None = None
    usable: bool = False
    reason: str = "no artifact was discovered for this layer"
    provenance: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def lens_inventory(
    rows: Sequence[BandLensRow | Mapping],
    *,
    layer_range: tuple[int, int],
    required_scale: int | None = None,
) -> dict:
    """The Stage-0 table: one row for **every** physical layer in the range.

    A layer with no discovered artifact still gets a row, marked unusable with
    the reason. That is the point of the table: the reader sees the whole
    window and can tell a confirmed lens from an absent one at a glance,
    without the notebook having to invent a path or a checksum for the gaps.

    Args:
        rows: Discovered rows. Extra layers outside ``layer_range`` are kept in
            the record but never enter ``usable_layers``.
        required_scale: When set, a row whose ``fit_scale`` differs is forced
            unusable — layers fitted at different scales are not one lens.
    """
    low, high = int(layer_range[0]), int(layer_range[1])
    if high < low:
        raise BandDesignRefused(f"empty layer range [{low}, {high}]")

    supplied: dict[int, BandLensRow] = {}
    for row in rows:
        record = row if isinstance(row, BandLensRow) else BandLensRow(**dict(row))
        if record.layer in supplied:
            raise BandDesignRefused(
                f"layer {record.layer} was offered twice in the lens inventory; "
                "which artifact a band rests on is not decided by sort order"
            )
        supplied[int(record.layer)] = record

    table: list[BandLensRow] = []
    for layer in range(low, high + 1):
        record = supplied.get(layer, BandLensRow(layer=layer))
        if record.usable and required_scale is not None and (
            record.fit_scale is None or int(record.fit_scale) != int(required_scale)
        ):
            record = BandLensRow(
                **{
                    **record.to_dict(),
                    "usable": False,
                    "reason": (
                        f"fitted at scale {record.fit_scale!r}, this band requires "
                        f"scale {required_scale}; layers fitted at different scales "
                        "are not one lens"
                    ),
                }
            )
        table.append(record)
    for layer in sorted(set(supplied) - set(range(low, high + 1))):
        table.append(supplied[layer])

    usable = tuple(
        row.layer for row in table if row.usable and low <= row.layer <= high
    )
    unusable = tuple(
        row.layer for row in table if not row.usable and low <= row.layer <= high
    )
    payload = {
        "version": BAND_DESIGN_VERSION,
        "layer_range": [low, high],
        "required_scale": None if required_scale is None else int(required_scale),
        "rows": [row.to_dict() for row in table],
        "usable_layers": list(usable),
        "unusable_layers": list(unusable),
        "contiguous_usable_runs": [list(run) for run in contiguous_runs(usable)],
        "n_usable": len(usable),
        "paths_and_checksums_are_read_from_artifacts": True,
    }
    return {**payload, "inventory_digest": payload_checksum(payload)}


def format_lens_inventory(inventory: Mapping) -> str:
    """The block printed before any model is loaded."""
    low, high = inventory["layer_range"]
    lines = [
        "=" * 100,
        f"J-LENS INVENTORY FOR THE BAND WINDOW L{low}-L{high} — read from artifacts, "
        "nothing assumed",
        "=" * 100,
        f"  {'layer':>5}  {'scale':>6}  {'usable':>6}  {'confirmation':<28} {'checksum':<24} path",
    ]
    for row in inventory["rows"]:
        checksum = row["lens_checksum"] or "-"
        lines.append(
            f"  {row['layer']:>5}  {str(row['fit_scale'] or '-'):>6}  "
            f"{('YES' if row['usable'] else 'no'):>6}  "
            f"{str(row['confirmation_verdict'] or '-'):<28} "
            f"{checksum[:24]:<24} {row['lens_path'] or '-'}"
        )
        if not row["usable"]:
            lines.append(f"         reason: {row['reason']}")
        elif row["fit_corpus"] or row["validation_set"]:
            lines.append(
                f"         corpus: {row['fit_corpus'] or '-'}   "
                f"validation: {row['validation_set'] or '-'}"
            )
    runs = ", ".join(f"{a}-{b}" for a, b in inventory["contiguous_usable_runs"]) or "none"
    lines += [
        "",
        f"  usable layers            {inventory['usable_layers']}",
        f"  contiguous usable runs   {runs}",
        f"  inventory digest         {inventory['inventory_digest']}",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------- the design


def band_design_record(
    *,
    inventory: Mapping,
    primary_band: LayerBand,
    suffix_bands: Sequence[LayerBand],
    position_rule: str = PRIMARY_POSITION_RULE,
    alphas: Sequence[float] = (PRIMARY_ALPHA, SECONDARY_ALPHA),
    conditions: Sequence[str] = BAND_CONDITIONS,
    arms: Sequence[str] = BAND_ARMS,
) -> dict:
    """The frozen band design, checksummed before any model result exists."""
    unknown = [name for name in conditions if name not in BAND_CONDITIONS]
    if unknown:
        raise BandDesignRefused(f"unknown band conditions {unknown}")
    unknown_arms = [name for name in arms if name not in BAND_ARMS]
    if unknown_arms:
        raise BandDesignRefused(f"unknown arms {unknown_arms}")
    for band in (primary_band, *suffix_bands):
        assert_contiguous(band.layers, what="predeclared band")
    payload = {
        "version": BAND_DESIGN_VERSION,
        "method_family": BAND_INTERVENTION_FAMILY,
        "study_version": BAND_SWAP_VERSION,
        "primary_band": primary_band.to_dict(),
        "suffix_bands": [band.to_dict() for band in suffix_bands],
        "band_keys": [band_key(band) for band in suffix_bands],
        "band_start_layers": [band.start for band in suffix_bands],
        "band_end_layer": primary_band.end,
        "every_physical_layer_in_each_band_is_patched": True,
        "hooks_installed_simultaneously_across_the_band": True,
        "coordinates_recomputed_per_layer_from_its_own_activation": True,
        "position_rule": str(position_rule),
        "alphas": [float(value) for value in alphas],
        "alpha_roles": dict(ALPHA_ROLES),
        "alpha_swept_per_sample": False,
        "conditions": list(conditions),
        "arms": list(arms),
        "usable_layers": list(inventory["usable_layers"]),
        "inventory_digest": inventory["inventory_digest"],
        "sampled_grid_is_not_a_band": (
            "[32, 35, 38, 40] is four confirmed layers, not the range 32-40; "
            "assert_contiguous refuses it"
        ),
        "single_layer_v2_run_is_untouched": True,
        "involution_does_not_forbid_a_band": (
            "the involution cancels one fixed update applied twice; each band "
            "layer has its own W_U @ J_l and re-reads its own coordinates"
        ),
    }
    return {**payload, "design_digest": payload_checksum(payload)}


def band_swap_fingerprint(
    *,
    design: Mapping,
    lens_checksums: Mapping[int, str],
    model_repo_id: str,
    model_revision: str,
    processor_revision: str,
    transformers_version: str,
    audio_protocol_fingerprint: str | None,
    prompt_protocol: Sequence[Mapping] | Mapping | None,
    directed_pairs: Sequence[Mapping],
    sample_identities: Mapping,
    thresholds: Mapping,
    coordinate_swap_method_version: str,
    fitting_manifest: Mapping | None = None,
    validation_manifest: Mapping | None = None,
    scoring_rule: str,
    verdict_version: str = BAND_VERDICT_VERSION,
) -> dict:
    """Everything a band result is bound to. A change here refuses a resume.

    Raises:
        BandDesignRefused: If any band layer has no recorded lens checksum. A
            patched layer that cannot name the lens its coordinates came from
            is not a measurement.
    """
    checksums = {int(layer): str(value) for layer, value in lens_checksums.items()}
    bands = [list(band["layers"]) for band in design["suffix_bands"]]
    missing = sorted(
        {layer for band in bands for layer in band if int(layer) not in checksums}
    )
    if missing:
        raise BandDesignRefused(
            f"no lens checksum recorded for band layer(s) {missing}; every "
            "patched layer must name the lens that defined its coordinates"
        )
    payload = {
        "intervention_family": BAND_INTERVENTION_FAMILY,
        "band_study_version": BAND_SWAP_VERSION,
        "coordinate_swap_method_version": str(coordinate_swap_method_version),
        "design_digest": design["design_digest"],
        "ordered_bands": bands,
        "band_keys": list(design["band_keys"]),
        "position_rule": design["position_rule"],
        "alphas": list(design["alphas"]),
        "alpha_roles": dict(design["alpha_roles"]),
        "conditions": list(design["conditions"]),
        "arms": list(design["arms"]),
        "lens_checksums_by_layer": {
            str(layer): checksums[layer] for layer in sorted(checksums)
        },
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
        "directed_pairs": [dict(pair) for pair in directed_pairs],
        "sample_identities": dict(sample_identities),
        "fitting_manifest": dict(fitting_manifest or {}),
        "validation_manifest": dict(validation_manifest or {}),
        "behavioral_scoring_rule": str(scoring_rule),
        "thresholds": dict(thresholds),
        "verdict_version": str(verdict_version),
    }
    return {**payload, "band_fingerprint_digest": payload_checksum(payload)}


# ------------------------------------------------------------- aggregation


@dataclass(frozen=True)
class BandSwapThresholds:
    """Frozen pass criteria for one band/arm/condition/modality/pair cell.

    Attributes:
        min_images: Independent photographs (or recordings of them) required.
        min_target_top1_rate: The **restricted-candidate** rate — the fraction
            of trials in which the downstream target answer outranks the other
            predeclared candidates. Not a global-vocabulary argmax rate, and not
            Anthropic's endpoint.
        control_top1_margin: How far the primary rate must exceed each matched
            control's rate. Zero means "strictly greater".
        max_identity_flip_rate_answer_arm: The answer arm exchanges the answer
            coordinates; if it also replaces the identity, it is not the
            control it claims to be.
        bootstrap_iterations / bootstrap_seed: The image-level bootstrap, fixed
            so an interval is reproducible from the stored cells alone.
    """

    min_images: int = 4
    min_target_top1_rate: float = 0.50
    control_top1_margin: float = 0.0
    max_identity_flip_rate_answer_arm: float = 0.25
    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 20260811
    blocking_controls: tuple[str, ...] = ("zero", "random", "unrelated")

    def __post_init__(self) -> None:
        if self.min_images < 2:
            raise ValueError("min_images must be at least 2")
        for name in ("min_target_top1_rate", "max_identity_flip_rate_answer_arm"):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not self.blocking_controls:
            raise ValueError("at least one blocking control is required")
        if self.bootstrap_iterations < 100:
            raise ValueError("bootstrap_iterations must be at least 100")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["blocking_controls"] = list(self.blocking_controls)
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum({"version": BAND_VERDICT_VERSION, **self.to_dict()})


def controls_for_condition(
    condition: str, *, blocking_controls: Sequence[str] = ("zero", "random", "unrelated")
) -> tuple[str, ...]:
    """The intensity-matched control conditions for one primary condition.

    ``swap_alpha2`` is compared against ``random_alpha2`` and
    ``unrelated_alpha2`` — never against the alpha=1 controls, which would
    compare a double-strength intervention with a single-strength baseline and
    credit the difference to the swap.
    """
    if condition not in ("swap_alpha1", "swap_alpha2"):
        raise BandDesignRefused(
            f"{condition!r} is not a primary condition; controls are defined for "
            "swap_alpha1 and swap_alpha2"
        )
    suffix = "alpha2" if condition.endswith("alpha2") else "alpha1"
    out = []
    for control in blocking_controls:
        out.append(control if control == "zero" else f"{control}_{suffix}")
    return tuple(out)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def bootstrap_interval(
    values: Sequence[float],
    *,
    iterations: int = 2000,
    seed: int = 20260811,
    lower: float = 2.5,
    upper: float = 97.5,
    label: str = "",
) -> dict:
    """Deterministic image-level percentile bootstrap of the mean.

    Resampling is done with a seeded, label-salted LCG rather than numpy so the
    interval is reproducible from the stored cell on any machine, with or
    without numpy, and so two different cells never share a resampling pattern.
    """
    observations = [float(value) for value in values]
    n = len(observations)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "lower": None,
            "upper": None,
            "iterations": int(iterations),
            "seed": int(seed),
            "percentiles": [lower, upper],
        }
    salt = int.from_bytes(
        hashlib.sha256(f"{seed}|{label}".encode()).digest()[:8], "big"
    )
    state = (salt ^ int(seed)) % (2**63 - 1) or 1
    means: list[float] = []
    for _ in range(int(iterations)):
        total = 0.0
        for _ in range(n):
            state = (6364136223846793005 * state + 1442695040888963407) % (2**64)
            total += observations[(state >> 33) % n]
        means.append(total / n)
    means.sort()

    def percentile(point: float) -> float:
        if n == 0:  # pragma: no cover - guarded above
            return math.nan
        index = min(
            len(means) - 1, max(0, int(round((point / 100.0) * (len(means) - 1))))
        )
        return means[index]

    return {
        "n": n,
        "mean": _mean(observations),
        "lower": percentile(lower),
        "upper": percentile(upper),
        "iterations": int(iterations),
        "seed": int(seed),
        "percentiles": [lower, upper],
    }


def band_trial_record(
    result: Mapping,
    *,
    band: Sequence[int],
    arm: str,
    condition: str,
    modality: str,
    source: str,
    target: str,
    source_answer: str,
    target_answer: str,
    readout: str,
    group_id: str,
    image_id: str,
    prompt_hash: str | None = None,
) -> dict:
    """One stored band trial, with the primary and secondary endpoints together.

    Built from :func:`jlens.mmpilot.coordinate_swap.run_swap_condition`'s own
    output so the top-1 decision, the rank, the log-prob and the margin all
    come from one scored forward pass and cannot drift apart.

    Raises:
        BandDesignRefused: If the layers the hooks actually fired at are not
            exactly the requested band. A trial that silently patched fewer
            layers than it claims is not a band trial.
    """
    layers = assert_contiguous(band, what="trial band")
    patched = tuple(int(layer) for layer in result.get("layers_patched", ()))
    if patched != layers:
        raise BandDesignRefused(
            f"the band {list(layers)} was requested but hooks fired at "
            f"{list(patched)}; every physical layer of a band is patched or the "
            "trial is not a band trial"
        )
    scores = dict(result["candidate_scores"])
    ranking = sorted(scores, key=lambda name: (-scores[name]["sum_logprob"], name))
    return {
        "status": "complete",
        "group_id": str(group_id),
        "image_id": str(image_id),
        "prompt_hash": prompt_hash,
        "band_key": band_key(layers),
        "band": list(layers),
        "band_start": int(layers[0]),
        "band_end": int(layers[-1]),
        "arm": str(arm),
        "condition": str(condition),
        "alpha": float(result["alpha"]),
        "alpha_is_extrapolation": bool(result["alpha_is_extrapolation"]),
        "modality": str(modality),
        "source": str(source),
        "target": str(target),
        "source_answer": str(source_answer),
        "target_answer": str(target_answer),
        "readout": str(readout),
        "prediction": result["prediction"],
        "clean_prediction": result["clean_prediction"],
        "target_rank": ranking.index(str(target_answer)) + 1,
        "n_candidates_scored": len(ranking),
        "candidate_ranking": ranking,
        "target_score": float(result["target_score"]),
        "clean_target_score": float(result["clean_target_score"]),
        "target_margin": float(result["target_margin"]),
        "target_margin_change": float(result["target_margin_change"]),
        "n_positions_patched": result["n_positions_patched"],
        "n_candidate_positions_skipped": result["n_candidate_positions_skipped"],
        "layers_patched": list(patched),
        "intervention_family": BAND_INTERVENTION_FAMILY,
        "candidate_scores": result["candidate_scores"],
    }


def read_band_units(run_dir) -> tuple[list[dict], dict]:
    """Re-read a completed band run's intervention units, read-only.

    Stage 4 is a CPU stage over units stage 3 already paid for, so it must be
    able to open a finished run directory without a model, a processor or the
    media. Each unit is validated against **its own** recorded checksum — a
    torn file is counted and dropped rather than aggregated — and the run's
    report supplies the predeclared bands and directed pairs, so the analysis
    is over the design that was frozen rather than over whatever happens to be
    on disk.

    Raises:
        BandDesignRefused: If the directory holds no report, no units, or a
            report from a different study.
    """
    import json
    from pathlib import Path

    root = Path(run_dir)
    report_path = root / "anthropic_band_swap_report.json"
    if not report_path.is_file():
        raise BandDesignRefused(
            f"{root} has no anthropic_band_swap_report.json; stage 4 analyses a "
            "completed band run, and the report is what states the bands and the "
            "directed pairs that were predeclared"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("intervention_family") != BAND_INTERVENTION_FAMILY:
        raise BandDesignRefused(
            f"{report_path} belongs to intervention family "
            f"{report.get('intervention_family')!r}, not {BAND_INTERVENTION_FAMILY!r}"
        )

    records: list[dict] = []
    invalid: list[str] = []
    for path in sorted((root / "units" / "intervention").glob("*.json")):
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            payload = stored["payload"]
            valid = stored.get("unit_checksum") == payload_checksum(payload)
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            valid, payload = False, None
        if not valid:
            invalid.append(str(path))
            continue
        if payload.get("status") == "complete":
            records.append(payload)
    if not records:
        raise BandDesignRefused(f"{root} holds no complete intervention units")
    context = {
        "run_dir": str(root),
        "report_checksum": report.get("report_checksum"),
        "band_keys": list((report.get("design") or {}).get("band_keys") or ()),
        "directed_pairs": list(report.get("directed_pairs") or ()),
        "thresholds": dict(report.get("thresholds") or {}),
        "n_units": len(records),
        "n_invalid_units": len(invalid),
        "invalid_units": invalid,
    }
    return records, context


def _cell_key(row: Mapping) -> tuple:
    return (
        str(row["band_key"]),
        str(row["arm"]),
        str(row["condition"]),
        str(row["modality"]),
        str(row["source"]),
        str(row["target"]),
        str(row["readout"]),
    )


def summarize_band_cells(
    records: Sequence[Mapping],
    *,
    thresholds: BandSwapThresholds | None = None,
) -> list[dict]:
    """Aggregate stored per-image band conditions, one row per image per cell.

    Every stored record must carry the top-1 prediction, the target answer, the
    target's rank in the scored candidate set, its log-prob, and the margin
    change against the clean run — so the primary endpoint and every secondary
    statistic come from the same unit and cannot drift apart.

    Raises:
        BandDesignRefused: If a cell contains two records for one photograph.
            Two trials on one image are not two independent observations.
    """
    thresholds = thresholds or BandSwapThresholds()
    grouped: dict[tuple, list[Mapping]] = {}
    for row in records:
        grouped.setdefault(_cell_key(row), []).append(row)

    out: list[dict] = []
    for key, rows in sorted(grouped.items()):
        images = {str(row["image_id"]) for row in rows}
        if len(images) != len(rows):
            raise BandDesignRefused(
                f"cell {key} holds {len(rows)} records over {len(images)} "
                "photograph(s); image-level aggregation would pseudoreplicate"
            )
        top1 = [
            1.0 if str(row["prediction"]) == str(row["target_answer"]) else 0.0
            for row in rows
        ]
        margins = [float(row["target_margin_change"]) for row in rows]
        partial = [
            1.0
            if (
                float(row["target_margin_change"]) > 0.0
                and str(row["prediction"]) != str(row["target_answer"])
            )
            else 0.0
            for row in rows
        ]
        ranks = [
            float(row["target_rank"])
            for row in rows
            if row.get("target_rank") is not None
        ]
        out.append(
            {
                "band_key": key[0],
                "band": list(rows[0].get("band", [])),
                "band_start": int(rows[0]["band_start"]),
                "arm": key[1],
                "condition": key[2],
                "alpha": float(rows[0]["alpha"]),
                "modality": key[3],
                "source": key[4],
                "target": key[5],
                "readout": key[6],
                "n_images": len(images),
                "image_ids": sorted(images),
                # --- the primary endpoint
                "target_top1_rate": _mean(top1),
                "target_top1_image_ids": sorted(
                    str(row["image_id"])
                    for row in rows
                    if str(row["prediction"]) == str(row["target_answer"])
                ),
                "target_top1_bootstrap": bootstrap_interval(
                    top1,
                    iterations=thresholds.bootstrap_iterations,
                    seed=thresholds.bootstrap_seed,
                    label="|".join(str(part) for part in key),
                ),
                # --- secondary movement statistics, never a success on their own
                "mean_target_rank": _mean(ranks) if ranks else None,
                "mean_target_logprob": _mean(
                    [float(row["target_score"]) for row in rows]
                ),
                "mean_target_margin": _mean(
                    [float(row["target_margin"]) for row in rows]
                ),
                "mean_target_margin_change": _mean(margins),
                "partial_movement_rate": _mean(partial),
                "partial_movement_label": PARTIAL_MOVEMENT,
                "clean_source_top1_rate": _mean(
                    [
                        1.0
                        if str(row["clean_prediction"]) == str(row["source_answer"])
                        else 0.0
                        for row in rows
                    ]
                ),
                "primary_endpoint": "target answer is top-1 of the scored candidates",
            }
        )
    return out


def _index(cells: Sequence[Mapping]) -> dict[tuple, Mapping]:
    return {_cell_key(row): row for row in cells}


def _direction_cell(
    index: Mapping[tuple, Mapping],
    *,
    band: str,
    arm: str,
    condition: str,
    modality: str,
    source: str,
    target: str,
    thresholds: BandSwapThresholds,
) -> dict:
    """One directed, modality-specific band cell, judged clause by clause."""
    primary = index.get((band, arm, condition, modality, source, target, PRIMARY_READOUT))
    diagnostic = index.get(
        (band, arm, condition, modality, source, target, DIAGNOSTIC_READOUT)
    )
    clauses: dict[str, bool] = {
        "downstream_property_record_present": primary is not None,
    }
    row = {
        "band_key": band,
        "arm": arm,
        "condition": condition,
        "modality": modality,
        "source": source,
        "target": target,
        "n_images": None if primary is None else int(primary["n_images"]),
        "target_top1_rate": None if primary is None else float(primary["target_top1_rate"]),
        "target_top1_interval": None if primary is None else primary["target_top1_bootstrap"],
        "mean_target_rank": None if primary is None else primary["mean_target_rank"],
        "mean_target_logprob": None if primary is None else primary["mean_target_logprob"],
        "mean_target_margin_change": (
            None if primary is None else float(primary["mean_target_margin_change"])
        ),
        "partial_movement_rate": (
            None if primary is None else float(primary["partial_movement_rate"])
        ),
        "identity_diagnostic": (
            None
            if diagnostic is None
            else {
                "identity_top1_rate": float(diagnostic["target_top1_rate"]),
                "n_images": int(diagnostic["n_images"]),
                "role": "intervention integrity only; never a reasoning success",
            }
        ),
        "control_comparisons": {},
    }
    if primary is None:
        row["clauses"] = clauses
        row["failed_clauses"] = sorted(clauses)
        row["passed"] = False
        return row

    clauses["minimum_independent_images"] = (
        int(primary["n_images"]) >= thresholds.min_images
    )
    clauses["target_answer_top1_rate"] = (
        float(primary["target_top1_rate"]) >= thresholds.min_target_top1_rate
    )
    for control in controls_for_condition(
        condition, blocking_controls=thresholds.blocking_controls
    ):
        control_cell = index.get(
            (band, arm, control, modality, source, target, PRIMARY_READOUT)
        )
        passed = control_cell is not None and (
            float(primary["target_top1_rate"])
            > float(control_cell["target_top1_rate"]) + thresholds.control_top1_margin
        )
        clauses[f"beats_{control}"] = passed
        row["control_comparisons"][control] = {
            "present": control_cell is not None,
            "primary_top1_rate": float(primary["target_top1_rate"]),
            "control_top1_rate": (
                None if control_cell is None else float(control_cell["target_top1_rate"])
            ),
            "alpha_matched": (
                None if control_cell is None else float(control_cell["alpha"])
            ),
            "passed": passed,
        }
    if arm == "answer":
        # The answer arm is the intermediate arm's control, and it is only that
        # if it left the identity alone. A missing identity record cannot pass
        # the clause by being absent.
        clauses["identity_record_present"] = diagnostic is not None
        clauses["identity_preserved_by_the_answer_arm"] = diagnostic is not None and (
            float(diagnostic["target_top1_rate"])
            <= thresholds.max_identity_flip_rate_answer_arm
        )

    row["clauses"] = clauses
    row["failed_clauses"] = sorted(name for name, ok in clauses.items() if not ok)
    row["passed"] = bool(clauses) and all(clauses.values())
    return row


def band_reasoning_verdict(
    cells: Sequence[Mapping],
    *,
    bands: Sequence[str],
    directed_pairs: Sequence[Mapping],
    modalities: Sequence[str],
    paper_comparable_modality: str = "text",
    thresholds: BandSwapThresholds | None = None,
) -> dict:
    """The restricted-candidate preference endpoint, per band/arm/condition/modality.

    Two results are produced and never merged:

    ``restricted_candidate_preference``
        Text-only: the downstream target answer outranks the other predeclared
        candidates after the intermediate swap, at alpha=1, beating every
        intensity-matched control. This is a **forced-choice preference over the
        supplied candidate set**, not a statement about what the model would
        emit. It is not comparable to Anthropic's global-argmax trial definition
        and must not be labelled paper-comparable; the corresponding
        unrestricted measurement is
        :func:`jlens.mmpilot.full_vocabulary.score_unrestricted_next_token`.

    ``modality_extension``
        The same criterion applied to image and spoken-audio evidence, reported
        separately per modality, plus a tri-modal conjunction that is labelled
        as **our extension** and is never a precondition for the text result.

    ``paper_comparable`` is retained as a **deprecated alias** of
    ``restricted_candidate_preference`` so completed reports and their readers
    keep resolving; its payload carries ``is_paper_comparable = False``.

    Identity replacement appears only as a diagnostic. A cell in which the
    identity flipped but the downstream answer did not is not a success, and a
    cell whose margin improved without reaching candidate-set top-1 is counted
    as :data:`PARTIAL_MOVEMENT`.
    """
    thresholds = thresholds or BandSwapThresholds()
    index = _index(cells)
    conditions = ("swap_alpha1", "swap_alpha2")
    direction_cells: list[dict] = []
    by_condition: dict[str, dict] = {}

    for condition in conditions:
        arms: dict[str, dict] = {}
        for arm in BAND_ARMS:
            band_rows: dict[str, dict] = {}
            for band in bands:
                rows = [
                    _direction_cell(
                        index,
                        band=str(band),
                        arm=arm,
                        condition=condition,
                        modality=str(modality),
                        source=str(pair["source"]),
                        target=str(pair["target"]),
                        thresholds=thresholds,
                    )
                    for pair in directed_pairs
                    for modality in modalities
                ]
                direction_cells.extend(rows)
                pair_pass = {
                    f"{pair['source']}->{pair['target']}": {
                        str(modality): any(
                            row["passed"]
                            and row["modality"] == str(modality)
                            and row["source"] == str(pair["source"])
                            and row["target"] == str(pair["target"])
                            for row in rows
                        )
                        for modality in modalities
                    }
                    for pair in directed_pairs
                }
                paper_pass = any(
                    row[str(paper_comparable_modality)]
                    for row in pair_pass.values()
                    if str(paper_comparable_modality) in row
                )
                tri_modal = any(
                    all(row.values()) for row in pair_pass.values()
                )
                band_rows[str(band)] = {
                    "band_key": str(band),
                    "band_start": int(str(band).split("-")[0]),
                    "paper_comparable_passed": bool(paper_pass),
                    "tri_modal_extension_passed": bool(tri_modal),
                    "per_modality_passed": {
                        str(modality): any(
                            row[str(modality)] for row in pair_pass.values()
                        )
                        for modality in modalities
                    },
                    "same_direction_pair_pass": pair_pass,
                    "n_direction_cells_passing": sum(
                        1 for row in rows if row["passed"]
                    ),
                    "n_direction_cells": len(rows),
                }
            arms[arm] = band_rows
        by_condition[condition] = arms

    def passing_bands(condition: str, arm: str, key: str) -> list[str]:
        return [
            band
            for band in bands
            if by_condition[condition][arm][str(band)][key]
        ]

    paper_bands = passing_bands("swap_alpha1", "intermediate", "paper_comparable_passed")
    extension_bands = passing_bands(
        "swap_alpha1", "intermediate", "tri_modal_extension_passed"
    )
    sensitivity_bands = passing_bands(
        "swap_alpha2", "intermediate", "paper_comparable_passed"
    )

    if paper_bands:
        verdict = "BAND_SWAP_PAPER_COMPARABLE_GO"
    elif sensitivity_bands:
        verdict = "BAND_SWAP_ALPHA2_ONLY"
    else:
        verdict = "BAND_SWAP_NO_GO"

    payload = {
        "version": BAND_VERDICT_VERSION,
        "study_version": BAND_SWAP_VERSION,
        "verdict": verdict,
        "primary_alpha": PRIMARY_ALPHA,
        "secondary_alpha": SECONDARY_ALPHA,
        "alpha_roles": dict(ALPHA_ROLES),
        "primary_endpoint": (
            "fraction of trials in which the target-appropriate downstream "
            "answer is top-1 of the externally scored candidate set"
        ),
        "endpoint_class": ENDPOINT_RESTRICTED_CANDIDATE,
        "full_vocabulary_evaluated": False,
        "identity_role": "intervention-integrity diagnostic only",
        "restricted_candidate_preference": {
            "modality": str(paper_comparable_modality),
            "endpoint_class": ENDPOINT_RESTRICTED_CANDIDATE,
            "criterion": (
                "restricted-candidate preference at alpha=1: the target answer "
                "outranks the other predeclared candidates by teacher-forced "
                "conditional sequence likelihood"
            ),
            "candidate_universe": "predeclared, externally supplied",
            "is_paper_comparable": False,
            "is_the_model_output": False,
            "passing_bands": paper_bands,
            "passed": bool(paper_bands),
        },
        "modality_extension": {
            "modalities": [str(modality) for modality in modalities],
            "tri_modal_conjunction_is_our_extension": True,
            "tri_modal_passing_bands": extension_bands,
            "per_band": {
                str(band): by_condition["swap_alpha1"]["intermediate"][str(band)][
                    "per_modality_passed"
                ]
                for band in bands
            },
        },
        "alpha2_sensitivity": {
            "role": ALPHA_ROLES["2.0"],
            "passing_bands": sensitivity_bands,
            "controls_are_alpha2_matched": True,
        },
        "tested_bands": [str(band) for band in bands],
        "by_condition": by_condition,
        "direction_cells": direction_cells,
        "thresholds": thresholds.to_dict(),
        "threshold_digest": thresholds.digest,
        "margin_only_counts_as": PARTIAL_MOVEMENT,
        "identity_success_alone_is_never_a_reasoning_go": True,
    }
    # Deprecated alias. Completed reports and their readers address this result
    # as ``paper_comparable``; the name is wrong and the payload says so, but
    # removing the key would break the readers of an immutable report.
    payload["paper_comparable"] = {
        **payload["restricted_candidate_preference"],
        "deprecated_name": True,
        "superseded_by": "restricted_candidate_preference",
    }
    return {**payload, "verdict_digest": payload_checksum(payload)}


def band_onset_timing(
    verdict: Mapping,
    *,
    bands: Sequence[str],
    directed_pairs: Sequence[Mapping],
    modalities: Sequence[str],
    condition: str = "swap_alpha1",
    modality: str = "text",
) -> dict:
    """Stage 4: when does each arm stop being able to change the answer?

    Both statistics are reported for both arms, and the classification uses
    ``deepest_effective_start`` — see the module docstring for why the earliest
    start is monotone-degenerate over nested suffix bands. Direction matching
    is required: the same ``source -> target`` pair must carry the intermediate
    and the answer comparison, so a strong forward direction in one arm cannot
    be paired with a strong reverse direction in the other.
    """
    by_condition = verdict["by_condition"]
    if condition not in by_condition:
        raise BandDesignRefused(
            f"the verdict carries no condition {condition!r}; it has "
            f"{sorted(by_condition)}"
        )
    ordered = sorted(bands, key=lambda value: int(str(value).split("-")[0]))

    def effective_starts(arm: str, pair: Mapping) -> list[int]:
        name = f"{pair['source']}->{pair['target']}"
        out = []
        for band in ordered:
            row = by_condition[condition][arm][str(band)]
            passes = row["same_direction_pair_pass"].get(name, {})
            if passes.get(str(modality)):
                out.append(int(str(band).split("-")[0]))
        return out

    pair_rows = []
    for pair in directed_pairs:
        arms = {arm: effective_starts(arm, pair) for arm in BAND_ARMS}
        earliest = {
            arm: (min(starts) if starts else None) for arm, starts in arms.items()
        }
        deepest = {
            arm: (max(starts) if starts else None) for arm, starts in arms.items()
        }
        intermediate, answer = deepest["intermediate"], deepest["answer"]
        if intermediate is not None and answer is not None:
            if intermediate < answer:
                classification = "INTERMEDIATE_CONSUMED_BEFORE_THE_ANSWER"
            elif intermediate == answer:
                classification = "SAME_TESTED_BAND_START"
            else:
                classification = "ANSWER_CONSUMED_FIRST"
        elif answer is not None:
            classification = "INTERMEDIATE_NULL_WITH_ANSWER_POSITIVE"
        elif intermediate is not None:
            classification = "ANSWER_NULL_WITH_INTERMEDIATE_POSITIVE"
        else:
            classification = "INCONCLUSIVE"
        pair_rows.append(
            {
                "pair": f"{pair['source']}->{pair['target']}",
                "effective_band_starts": arms,
                "earliest_effective_start": earliest,
                "deepest_effective_start": deepest,
                "classification": classification,
            }
        )

    classifications = {row["classification"] for row in pair_rows}
    if "INTERMEDIATE_CONSUMED_BEFORE_THE_ANSWER" in classifications:
        verdict_name = "BAND_ONSET_INTERMEDIATE_EARLIER"
    elif "SAME_TESTED_BAND_START" in classifications:
        verdict_name = "BAND_ONSET_SAME_TESTED_START"
    elif "ANSWER_CONSUMED_FIRST" in classifications:
        verdict_name = "BAND_ONSET_ANSWER_EARLIER"
    else:
        verdict_name = "BAND_ONSET_INCONCLUSIVE"

    payload = {
        "version": BAND_TIMING_VERSION,
        "verdict": verdict_name,
        "condition": str(condition),
        "modality": str(modality),
        "modalities_available": [str(name) for name in modalities],
        "tested_band_starts": [int(str(band).split("-")[0]) for band in ordered],
        "tested_bands": [str(band) for band in ordered],
        "pairs": pair_rows,
        "classified_on": "deepest_effective_start",
        "why_not_earliest": (
            "suffix bands are nested, so an arm carried by the deepest layers "
            "passes at every start and its earliest effective start is the "
            "earliest tested start by construction; both numbers are reported"
        ),
        "cross_arm_direction_matching_required": True,
        "band_starts_are_not_exact_physical_onsets": True,
        "native_direct_readout_convergence_gate_used": False,
        "source_derived_steering_used": False,
    }
    return {**payload, "timing_digest": payload_checksum(payload)}
