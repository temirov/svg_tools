#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "svgwrite>=1.4.3",
#   "fonttools>=4.44.0",
#   "svgpathtools>=1.6.1"
# ]
# ///
# Render any text into a **valid** single-path SVG.
# Fixes the previous “invalid value for attribute ‘d’” error by
# normalising the path-string format before saving.

import argparse
import re
from pathlib import Path

import svgwrite
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from svgpathtools import parse_path


# ------------------------------------------------------------ #
# Core helpers                                                 #
# ------------------------------------------------------------ #
def glyph_d(glyph_name: str, glyph_set) -> str:
    pen = SVGPathPen(glyph_set)
    glyph_set[glyph_name].draw(pen)
    return pen.getCommands()  # raw “d” for this glyph


def text_to_d(text: str, font_file: Path, font_size: int = 1024):
    """Return (d_attr, width, height) for the supplied text."""
    font = TTFont(str(font_file))
    glyph_set = font.getGlyphSet()
    cmap = font.getBestCmap()
    upm = font["head"].unitsPerEm
    ascent = font["hhea"].ascent
    descent = abs(font["hhea"].descent)
    hmtx = font["hmtx"].metrics

    scale = font_size / upm
    x_cursor = 0
    segments = []

    for ch in text:
        gname = cmap.get(ord(ch))
        if not gname:
            continue

        raw = glyph_d(gname, glyph_set)
        path_obj = (
            parse_path(raw)
            .translated(complex(x_cursor, ascent))  # shift baseline
            .scaled(scale, -scale)  # flip y-axis
        )
        segments.append(path_obj.d())
        x_cursor += hmtx[gname][0]  # advance-width

    width = x_cursor * scale
    height = (ascent + descent) * scale
    d_full = " ".join(segments)

    # --- NORMALISE “d” STRING ------------------------------------------- #
    # 1. ensure a space before EVERY command letter
    d_full = re.sub(r"([A-Za-z])", r" \1 ", d_full)
    # 2. collapse whitespace
    d_full = re.sub(r"\s+", " ", d_full).strip()
    # -------------------------------------------------------------------- #

    return d_full, width, height


def save_svg(d_attr: str, width: float, height: float, out_file: Path):
    dwg = svgwrite.Drawing(
        filename=str(out_file),
        size=(f"{width}px", f"{height}px"),
        viewBox=f"0 0 {width} {height}",
        profile="tiny",
    )
    dwg.add(
        dwg.path(
            d=d_attr,
            fill="none",
            stroke="black",
            stroke_width=8,
            stroke_linejoin="round",
            stroke_linecap="round",
        )
    )
    dwg.save()


# ------------------------------------------------------------ #
# CLI                                                          #
# ------------------------------------------------------------ #
def main():
    p = argparse.ArgumentParser(description="Convert text → single-path SVG.")
    p.add_argument("text", help="Text to render")
    p.add_argument("--font", required=True, help="TTF/OTF font file")
    p.add_argument("--out", required=True, help="Output SVG filename")
    p.add_argument("--size", type=int, default=1024, help="Font size UPM units")
    args = p.parse_args()

    d_attr, w, h = text_to_d(args.text, Path(args.font), args.size)
    save_svg(d_attr, w, h, Path(args.out))


if __name__ == "__main__":
    main()
