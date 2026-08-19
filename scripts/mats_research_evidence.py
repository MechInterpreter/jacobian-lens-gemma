"""Build source-backed tables and figures for the MATS research decision log.

The script reads only archived JSON reports. It never recomputes a scientific
result. Run it directly, or from notebooks/mats_research_evidence_analysis.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_ROOT = ROOT / "tmp" / "research_evidence_20260819"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "mats_application" / "figures"

INK = "#202124"
MUTED = "#5F6368"
GRID = "#DADCE0"
BLUE = "#1A73E8"
GREEN = "#188038"
RED = "#D93025"
AMBER = "#F9AB00"
PURPLE = "#9334E6"
PALE = "#F8F9FA"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _save(img: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, dpi=(180, 180))
    return str(path)


def _title(draw: ImageDraw.ImageDraw, text: str, subtitle: str = "") -> None:
    draw.text((72, 42), text, fill=INK, font=_font(34, True))
    if subtitle:
        draw.text((72, 88), subtitle, fill=MUTED, font=_font(19))


def _bar_chart(
    rows: list[tuple[str, float, str]],
    *,
    title: str,
    subtitle: str,
    path: Path,
    maximum: float = 1.0,
    threshold: float | None = None,
) -> str:
    width = 1500
    row_h = 58
    height = 170 + row_h * len(rows) + 70
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    _title(draw, title, subtitle)
    left, right, top = 350, 1400, 150
    if threshold is not None:
        x = left + int((right - left) * threshold / maximum)
        draw.line((x, top - 12, x, height - 58), fill=AMBER, width=4)
        draw.text((x + 8, top - 38), f"gate {threshold:.0%}", fill=AMBER, font=_font(17, True))
    for i, (label, value, color) in enumerate(rows):
        y = top + i * row_h
        draw.text((72, y + 6), label, fill=INK, font=_font(20))
        draw.rounded_rectangle((left, y, right, y + 34), 8, fill=PALE)
        x2 = left + int((right - left) * min(value, maximum) / maximum)
        draw.rounded_rectangle((left, y, x2, y + 34), 8, fill=color)
        draw.text((right + 16, y + 4), f"{value:.3f}", fill=INK, font=_font(19, True))
    draw.text((72, height - 42), "Values are read from archived run reports; no result is recomputed.", fill=MUTED, font=_font(16))
    return _save(img, path)


def build_figures(
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    evidence_root = Path(evidence_root)
    output_dir = Path(output_dir)

    audio = _read(
        evidence_root
        / "mmaudio_native_audio_transfer_20260806T144822"
        / "native_audio_transfer_summary.json"
    )
    band = _read(
        evidence_root
        / "bandcorr_real_eb5b00f135e4"
        / "artifacts"
        / "corrected_validation_v1"
        / "band_interior_corrected_validation_report.json"
    )
    l32 = _read(
        evidence_root
        / "mml32res_l32_convergence_resolution_20260810T174731"
        / "l32_convergence_resolution_summary.json"
    )
    alpha1 = _read(
        evidence_root
        / "mmalpha1confirm64_real_df0ce0404c32"
        / "alpha1_exact_swap_recruitment64_confirmation_report.json"
    )
    adjacent = _read(
        evidence_root / "mmpre_real_fdddd750b4c0" / "adjacent_lens_table.json"
    )

    figures: dict[str, str] = {}

    band_rows = []
    for row in band["band_verdict"]["layers"]:
        color = GREEN if row["confirmation_passed"] else RED
        band_rows.append((f"Layer {row['layer']}", row["mean_reciprocal_rank"], color))
    figures["lens_validation"] = _bar_chart(
        band_rows,
        title="Corrected J-lens confirmation across the L32–L40 window",
        subtitle="L32 failed the frozen gate; L33–L40 passed on one untouched confirmation population",
        path=output_dir / "figure_1_corrected_lens_validation.png",
    )

    cap = audio["verdicts"]["A_audio_capability"]["per_concept"]
    modality_colors = {"text": BLUE, "image": GREEN, "spoken_audio": PURPLE}
    capability_rows = []
    for concept in ["bird", "cat", "giraffe", "microwave", "toilet", "zebra"]:
        for modality in ["text", "image", "spoken_audio"]:
            capability_rows.append(
                (f"{concept} · {modality.replace('_', ' ')}", cap[concept][modality]["accuracy"], modality_colors[modality])
            )
    figures["capability"] = _bar_chart(
        capability_rows,
        title="Behavioral capability before causal analysis",
        subtitle="Restricted six-candidate accuracy, eight examples per concept × modality",
        path=output_dir / "figure_2_trimodal_capability.png",
        threshold=0.70,
    )

    endpoint_steps = [
        ("Candidate-conditioned tri-modal", "Controlled target log-probability", GREEN),
        ("Paper-style sparse-grid swap", "Restricted-candidate preference", AMBER),
        ("Validated L33–L40 band", "α=2 sensitivity only", AMBER),
        ("Full-vocabulary word endpoint", "No-go", RED),
        ("Digit endpoint, α=2", "No-go", RED),
        ("Exact coordinate exchange, α=1", "2/48 hidden-reasoning successes", RED),
        ("Direct-answer positive control", "24/24 bird→cat; 0/24 cat→bird", BLUE),
    ]
    width, height = 1600, 720
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    _title(draw, "Endpoint semantics changed what the evidence could claim", "Each pivot tightened the endpoint from conditional preference toward unrestricted next-token behavior")
    y = 155
    for i, (name, result, color) in enumerate(endpoint_steps, 1):
        draw.ellipse((76, y + 4, 112, y + 40), fill=color)
        draw.text((86, y + 8), str(i), fill="white", font=_font(17, True), anchor="ma")
        draw.text((138, y), name, fill=INK, font=_font(22, True))
        draw.text((760, y + 2), result, fill=color, font=_font(20, True))
        if i < len(endpoint_steps):
            draw.line((94, y + 42, 94, y + 78), fill=GRID, width=4)
        y += 76
    figures["endpoint_timeline"] = _save(img, output_dir / "figure_3_endpoint_pivot_timeline.png")

    cells = alpha1["aggregation"]["cells"]
    primary = [
        row for row in cells
        if row["condition"] == "swap_alpha1" and row["arm"] in {"intermediate", "answer"}
    ]
    width, height = 1600, 800
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    _title(draw, "Exact α=1 coordinate exchange: unrestricted digit outcomes", "Eight images per direction × modality cell; bars show observed greedy next-token success")
    groups = [(s, t, m) for s, t in [("bird", "cat"), ("cat", "bird")] for m in ["text", "image", "spoken_audio"]]
    left, right, top, bottom = 120, 1510, 170, 680
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    for tick in range(0, 9, 2):
        ytick = bottom - int((bottom - top) * tick / 8)
        draw.line((left, ytick, right, ytick), fill=GRID, width=2)
        draw.text((78, ytick), str(tick), fill=MUTED, font=_font(17), anchor="mm")
    slot = (right - left) / len(groups)
    for i, (source, target, modality) in enumerate(groups):
        x = left + i * slot + 24
        values = {}
        for row in primary:
            if row["source"] == source and row["target"] == target and row["modality"] == modality:
                values[row["arm"]] = row["successes"]
        for j, (arm, color) in enumerate([("intermediate", PURPLE), ("answer", BLUE)]):
            value = values.get(arm, 0)
            x1 = int(x + j * 66)
            y1 = bottom - int((bottom - top) * value / 8)
            draw.rectangle((x1, y1, x1 + 52, bottom), fill=color)
            draw.text((x1 + 26, y1 - 20), str(value), fill=color, font=_font(18, True), anchor="mm")
        label = f"{source}→{target}\n{modality.replace('_', ' ')}"
        draw.multiline_text((int(x + 58), bottom + 18), label, fill=INK, font=_font(16), anchor="ma", align="center", spacing=2)
    draw.rectangle((1100, 112, 1128, 140), fill=PURPLE)
    draw.text((1138, 112), "hidden intermediate arm", fill=INK, font=_font(18))
    draw.rectangle((1100, 146, 1128, 174), fill=BLUE)
    draw.text((1138, 146), "direct-answer control", fill=INK, font=_font(18))
    figures["alpha1"] = _save(img, output_dir / "figure_4_alpha1_unrestricted_outcomes.png")

    cls = l32["convergence"]["classification"]
    l32_rows = [
        ("Pooled clean-answer agreement", cls["pooled_clean_agreement_argmax"], BLUE),
        ("Pooled target accuracy", cls["pooled_target_accuracy_argmax"], PURPLE),
    ]
    figures["l32"] = _bar_chart(
        l32_rows,
        title="Independent L32 native-readout audit remained ambiguous",
        subtitle="Frozen bars: NOT_CONVERGED ≤ 0.50; CONVERGED ≥ 0.90 (all modalities required)",
        path=output_dir / "figure_5_l32_ambiguity.png",
        threshold=0.50,
    )

    summary = {
        "evidence_root": str(evidence_root),
        "figures": figures,
        "verified_results": {
            "audio_capability": audio["verdicts"]["A_audio_capability"]["verdict"],
            "representational_transfer": audio["verdicts"]["B_representational_transfer"]["verdict"],
            "corrected_band": band["band_verdict"]["verdict"],
            "corrected_band_passing_layers": band["band_verdict"]["layers_passing"],
            "l32_independent_classification": cls["classification"],
            "adjacent_j_and_r_lens": adjacent["verdict"],
            "alpha1_exact_swap": alpha1["verdict"]["verdict"],
        },
        "alpha1_primary_cells": primary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evidence_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    result = build_figures()
    print(json.dumps(result["verified_results"], indent=2))
    print("figures")
    for path in result["figures"].values():
        print(" ", path)
