# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Render the repository's editable SVG schematics to PNG.

The SVG is the source of truth: it is text, it diffs, and its equations can be
reviewed line by line. The PNG is a build product committed beside it so the
figure can be dropped into a document without a renderer. Never edit the PNG.

Usage::

    python -m pip install cairosvg
    python scripts/render_schematic.py                 # render everything
    python scripts/render_schematic.py docs/assets/intervention_methods.svg

``cairosvg`` is deliberately not a project dependency: nothing in ``jlens``
imports it, and the tests check the SVG's *content*, not the rasterizer's
output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS = REPO_ROOT / "docs" / "assets"

#: Rendered width in pixels. Fixed so a re-render is a no-op unless the SVG
#: actually changed.
OUTPUT_WIDTH = 1500


def render(svg_path: Path, *, width: int = OUTPUT_WIDTH) -> Path:
    """Rasterize one SVG next to itself as ``.png``. Returns the PNG path."""
    try:
        import cairosvg
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "cairosvg is required to render the schematic:\n"
            "    python -m pip install cairosvg\n"
            f"(the SVG source at {svg_path} is the artifact under version "
            "control; the PNG is a build product)"
        ) from error

    png_path = svg_path.with_suffix(".png")
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        output_width=width,
        background_color="white",
    )
    return png_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "svg",
        nargs="*",
        type=Path,
        help="SVG files to render (default: every SVG in docs/assets)",
    )
    parser.add_argument("--width", type=int, default=OUTPUT_WIDTH)
    arguments = parser.parse_args(argv)

    targets = arguments.svg or sorted(DEFAULT_ASSETS.glob("*.svg"))
    if not targets:
        print(f"no SVG files found under {DEFAULT_ASSETS}", file=sys.stderr)
        return 1
    for svg_path in targets:
        png_path = render(svg_path, width=arguments.width)
        print(f"{svg_path} -> {png_path} ({png_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
