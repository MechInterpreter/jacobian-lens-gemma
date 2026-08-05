# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The layer grid, the multilayer capture plan, and an honest budget.

The budget is derived from **one real measurement on this project's own
hardware**, not from a FLOP model: the completed pilot fitted 100 WikiText
prompts x 128 tokens x 7 source layers (shallowest = 3) in 9,665.4 s on one
NVIDIA L4 (``docs/pilot_report.md``). Everything else is an extrapolation from
it, and the uncertainty band is wide on purpose because a single measurement at
a different depth is thin evidence.

The cost model that extrapolation rests on comes from the upstream estimator
(``jlens/fitting.py``), not from guesswork:

* one forward pass per prompt, then ``ceil(d_model / dim_batch)`` backward
  passes against the retained graph — **320** for Gemma 4 E4B at
  ``dim_batch = 8``;
* every backward pass runs from the target layer down to the **shallowest**
  source layer and extracts *all* source layers at once
  (``torch.autograd.grad(inputs=source_activations)``).

So wall time scales with the **backward span**, not with the number of layers.
Adding a layer inside the existing span costs one slice-copy per backward pass.
Lowering the shallowest layer is the expensive change. Both facts are load
bearing for reading the numbers this module produces.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from jlens.mmpilot.store import payload_checksum

#: The predetermined physical layer grid. 20/26/32/38 are carried over from the
#: v2 recalibration and the completed localization run so results are directly
#: comparable; 8 and 14 extend genuinely earlier; 35 and 40 bracket the
#: established layer 38 so a pass there cannot be read as an isolated point.
CALIBRATION_LAYERS = (8, 14, 20, 26, 32, 35, 38, 40)

#: Nested scale points. See :mod:`jlens.calibration.scale` for the plateau rule.
SCALE_POINTS = (1_000, 5_000, 10_000)

#: Never reached automatically; requires its own switch and its own budget
#: confirmation.
OPTIONAL_SCALE_POINTS = (25_000, 50_000)

GEMMA4_N_LAYERS = 42
GEMMA4_D_MODEL = 2560
DEFAULT_TARGET_LAYER = 41

#: The one real measurement. Do not edit without a new measurement to replace it.
PILOT_MEASUREMENT = {
    "source": "docs/pilot_report.md",
    "device": "NVIDIA L4",
    "n_prompts": 100,
    "n_source_layers": 7,
    "shallowest_source_layer": 3,
    "target_layer": 41,
    "max_seq_len": 128,
    "dim_batch": 8,
    "total_seconds": 9665.4,
    "seconds_per_prompt": 96.654,
    "probe_peak_cuda_gb": 16.84,
    "probe_max_seq_len": 48,
    "full_fit_peak_memory_recorded": False,
}

#: Multiplicative band applied to the central estimate. The low end assumes the
#: span model slightly overstates savings; the high end covers per-layer copy
#: overhead, a slower L4 allocation, and Colab noise.
UNCERTAINTY_BAND = (0.95, 1.25)

#: Realistic uninterrupted Colab GPU session, for translating hours into
#: sessions. Deliberately conservative.
COLAB_SESSION_HOURS = 12.0


def normalized_depth(layer: int, *, n_layers: int = GEMMA4_N_LAYERS) -> int:
    """Depth as a percentage of the stack, the project's standard convention."""
    return round(100 * int(layer) / int(n_layers))


# ------------------------------------------------------------- capture plan


@dataclass(frozen=True)
class CapturePlan:
    """How the selected layers are captured, and what that costs per prompt.

    The important field is :attr:`layers_per_forward`: it is the whole layer
    grid, because the upstream estimator already differentiates to every source
    layer from a single forward pass and a single set of backward passes. There
    is no per-layer pass to plan.
    """

    layers: tuple[int, ...]
    target_layer: int
    d_model: int
    dim_batch: int
    max_seq_len: int
    skip_first: int
    n_layers: int = GEMMA4_N_LAYERS

    @property
    def shallowest_layer(self) -> int:
        return min(self.layers)

    @property
    def backward_span(self) -> int:
        """Blocks each backward pass traverses — the cost driver."""
        return self.target_layer - self.shallowest_layer

    @property
    def backward_passes_per_prompt(self) -> int:
        return math.ceil(self.d_model / self.dim_batch)

    @property
    def forwards_per_prompt(self) -> int:
        return 1

    @property
    def layers_per_forward(self) -> tuple[int, ...]:
        """Every configured layer. One pass captures all of them."""
        return self.layers

    @property
    def valid_positions_per_prompt(self) -> int:
        """Source positions entering the average at full sequence length."""
        return max(0, self.max_seq_len - self.skip_first - 1)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["layers"] = list(self.layers)
        payload.update(
            {
                "normalized_depth": [normalized_depth(l, n_layers=self.n_layers)
                                     for l in self.layers],
                "shallowest_layer": self.shallowest_layer,
                "backward_span": self.backward_span,
                "backward_passes_per_prompt": self.backward_passes_per_prompt,
                "forwards_per_prompt": self.forwards_per_prompt,
                "valid_positions_per_prompt": self.valid_positions_per_prompt,
                "all_layers_captured_in_one_pass": True,
                "per_layer_parameters": "one [d_model, d_model] matrix per layer, "
                                        "never shared",
            }
        )
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())


def build_capture_plan(
    *,
    layers: Sequence[int] = CALIBRATION_LAYERS,
    target_layer: int = DEFAULT_TARGET_LAYER,
    d_model: int = GEMMA4_D_MODEL,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = 16,
    n_layers: int = GEMMA4_N_LAYERS,
) -> CapturePlan:
    """Validate the grid and return the capture plan.

    Raises:
        ValueError: If the grid is empty, out of range, or reaches the target
            layer. Upstream enforces ``source < target`` too; catching it here
            means a bad grid fails before a 16 GB model load rather than after.
    """
    unique = sorted({int(layer) for layer in layers})
    if not unique:
        raise ValueError("the layer grid is empty")
    if unique[0] < 0 or unique[-1] >= n_layers:
        raise ValueError(
            f"layers {unique} out of range for a {n_layers}-layer model"
        )
    if unique[-1] >= target_layer:
        raise ValueError(
            f"every source layer must be shallower than target_layer={target_layer}; "
            f"got max={unique[-1]}"
        )
    if max_seq_len <= skip_first + 1:
        raise ValueError(
            f"max_seq_len={max_seq_len} leaves no valid positions with "
            f"skip_first={skip_first}"
        )
    if dim_batch <= 0:
        raise ValueError(f"dim_batch must be positive, got {dim_batch}")
    return CapturePlan(
        layers=tuple(unique),
        target_layer=int(target_layer),
        d_model=int(d_model),
        dim_batch=int(dim_batch),
        max_seq_len=int(max_seq_len),
        skip_first=int(skip_first),
        n_layers=int(n_layers),
    )


# ------------------------------------------------------------------- budget


def _seconds_per_prompt(plan: CapturePlan) -> float:
    """Central per-prompt estimate, scaled from the pilot by backward span."""
    anchor_span = PILOT_MEASUREMENT["target_layer"] - PILOT_MEASUREMENT[
        "shallowest_source_layer"
    ]
    return float(PILOT_MEASUREMENT["seconds_per_prompt"]) * (
        plan.backward_span / anchor_span
    )


@dataclass(frozen=True)
class Budget:
    """Compute and storage estimate for one capture plan and a scale schedule.

    Scale rows are **cumulative**, because the scale points are nested: reaching
    10,000 also produces the 1,000 and 5,000 lenses. Summing the rows would
    triple-count the same work.
    """

    plan_digest: str
    layers: tuple[int, ...]
    scale_points: tuple[int, ...]
    seconds_per_prompt: float
    seconds_per_prompt_low: float
    seconds_per_prompt_high: float
    per_scale: tuple[dict, ...]
    checkpoint_bytes: int
    snapshot_bytes_total: int
    published_bytes_max: int
    reports_bytes: int
    model_download_gb: float
    non_fitting_model_minutes: tuple[float, float]
    measurement: dict = field(default_factory=lambda: dict(PILOT_MEASUREMENT))

    @property
    def drive_bytes_total(self) -> int:
        return (
            self.checkpoint_bytes
            + self.snapshot_bytes_total
            + self.published_bytes_max
            + self.reports_bytes
        )

    @property
    def largest_scale(self) -> int:
        return max(self.scale_points)

    def row(self, scale: int) -> dict:
        for entry in self.per_scale:
            if entry["scale"] == int(scale):
                return entry
        raise KeyError(f"no budget row for scale {scale}")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["layers"] = list(self.layers)
        payload["scale_points"] = list(self.scale_points)
        payload["per_scale"] = [dict(entry) for entry in self.per_scale]
        payload["non_fitting_model_minutes"] = list(self.non_fitting_model_minutes)
        payload["drive_bytes_total"] = self.drive_bytes_total
        payload["drive_gib_total"] = round(self.drive_bytes_total / 2**30, 3)
        payload["cumulative_not_additive"] = True
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())


def estimate_budget(
    plan: CapturePlan,
    *,
    scale_points: Sequence[int] = SCALE_POINTS,
    n_validation: int = 128,
    n_confirmation: int = 128,
    session_hours: float = COLAB_SESSION_HOURS,
) -> Budget:
    """Extrapolate the pilot measurement to ``scale_points`` under ``plan``."""
    points = tuple(sorted(int(point) for point in scale_points))
    if not points:
        raise ValueError("estimate_budget needs at least one scale point")

    central = _seconds_per_prompt(plan)
    low = central * UNCERTAINTY_BAND[0]
    high = central * UNCERTAINTY_BAND[1]

    matrix_elements = plan.d_model * plan.d_model
    n_layers_fitted = len(plan.layers)
    checkpoint_bytes = n_layers_fitted * matrix_elements * 4  # fp32 running sum
    snapshot_bytes_each = n_layers_fitted * matrix_elements * 2  # fp16 lens

    rows: list[dict] = []
    for point in points:
        seconds = central * point
        rows.append(
            {
                "scale": point,
                "fitting_seconds_central": round(seconds, 1),
                "fitting_hours_central": round(seconds / 3600.0, 2),
                "fitting_hours_low": round(low * point / 3600.0, 2),
                "fitting_hours_high": round(high * point / 3600.0, 2),
                "colab_sessions_central": math.ceil(seconds / 3600.0 / session_hours),
                "colab_sessions_high": math.ceil(
                    high * point / 3600.0 / session_hours
                ),
                "forward_passes": point * plan.forwards_per_prompt,
                "backward_passes": point * plan.backward_passes_per_prompt,
                "cumulative": True,
                "incremental_prompts_from_previous": point - (rows[-1]["scale"] if rows else 0),
            }
        )

    return Budget(
        plan_digest=plan.digest,
        layers=plan.layers,
        scale_points=points,
        seconds_per_prompt=round(central, 2),
        seconds_per_prompt_low=round(low, 2),
        seconds_per_prompt_high=round(high, 2),
        per_scale=tuple(rows),
        checkpoint_bytes=checkpoint_bytes,
        snapshot_bytes_total=snapshot_bytes_each * len(points),
        published_bytes_max=snapshot_bytes_each,
        reports_bytes=20 * 1024 * 1024,
        model_download_gb=16.0,
        non_fitting_model_minutes=(
            round(2 + 2 * (n_validation + n_confirmation) / 128.0 * 5, 1),
            round(2 + 2 * (n_validation + n_confirmation) / 128.0 * 15, 1),
        ),
    )


def format_budget(budget: Budget, plan: CapturePlan) -> str:
    """The block printed before any real model work is allowed."""
    lines = [
        "COMPUTE AND STORAGE BUDGET — extrapolated from one measurement",
        f"  anchor              {PILOT_MEASUREMENT['n_prompts']} prompts x "
        f"{PILOT_MEASUREMENT['n_source_layers']} layers (shallowest "
        f"{PILOT_MEASUREMENT['shallowest_source_layer']}) in "
        f"{PILOT_MEASUREMENT['total_seconds']} s on an "
        f"{PILOT_MEASUREMENT['device']} ({PILOT_MEASUREMENT['source']})",
        f"  layers              {list(plan.layers)}  -> target L{plan.target_layer}",
        f"  backward span       {plan.backward_span} blocks "
        f"(anchor span {PILOT_MEASUREMENT['target_layer'] - PILOT_MEASUREMENT['shallowest_source_layer']})",
        f"  per prompt          1 forward + {plan.backward_passes_per_prompt} backward "
        f"passes (dim_batch={plan.dim_batch}, d_model={plan.d_model})",
        f"  seconds per prompt  {budget.seconds_per_prompt:.1f} "
        f"(range {budget.seconds_per_prompt_low:.1f}-{budget.seconds_per_prompt_high:.1f})",
        "",
        "  Scale points are NESTED: rows are cumulative, not additive.",
        "  scale        hours (central)   range            ~12h sessions",
    ]
    for row in budget.per_scale:
        lines.append(
            f"  {row['scale']:>7,}      {row['fitting_hours_central']:>8.1f}    "
            f"{row['fitting_hours_low']:>6.1f}-{row['fitting_hours_high']:<7.1f}  "
            f"{row['colab_sessions_central']}-{row['colab_sessions_high']}"
        )
    lines += [
        "",
        f"  checkpoint          {budget.checkpoint_bytes / 2**20:.0f} MiB (fp32 running sum)",
        f"  scale snapshots     {budget.snapshot_bytes_total / 2**20:.0f} MiB "
        f"({len(budget.scale_points)} x fp16)",
        f"  published (max)     {budget.published_bytes_max / 2**20:.0f} MiB",
        f"  Drive total         {budget.drive_bytes_total / 2**30:.2f} GiB",
        f"  model download      ~{budget.model_download_gb:.0f} GB per session "
        f"unless the HF cache lives on Drive",
        f"  non-fitting model work  {budget.non_fitting_model_minutes[0]:.0f}-"
        f"{budget.non_fitting_model_minutes[1]:.0f} min for the whole study",
        "",
        "  Fitting is the entire cost. Validation and confirmation are forward-pass",
        "  workloads; the 320 backward passes per prompt happen only during fitting.",
        "  A fit parallelizes exactly across runtimes via JacobianLens.merge().",
    ]
    return "\n".join(lines)
