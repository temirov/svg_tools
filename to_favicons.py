#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "cairosvg>=2.7.0",
#   "pillow>=10.2.0",
# ]
# ///

"""
favicon_packager.py

Generate a favicon package from a single SVG with two presets.

Presets
  minimal (default):
    - favicon.ico (16, 32, 48, 64)
    - favicon-32x32.png
    - apple-touch-icon.png (180)
    - android-chrome-192x192.png
    - android-chrome-512x512.png
    - safari-pinned-tab.svg
    - site.webmanifest
    - optional Windows tiles (when --windows-tiles)

  full:
    Everything in minimal, plus PNG sizes:
    16, 24, 32, 48, 72, 96, 128, 144, 152, 167, 256, 384

Output layout
  <folder>/<assets_path>/...
  - If --site-name is provided: <site-slug>/<assets_path>/...
  - If --site-name is omitted:  site/<assets_path>/...

Defaults
  assets_path = "assets/favicons"
  snippet URLs reference "/{assets_path}/..."
  manifest defaults: name/short_name = "My Site", theme_color = "#111111", background_color = "#ffffff"
"""

import argparse
import json
import os
import re
import shutil
from typing import Dict, List, Tuple

from PIL import Image
import cairosvg


def slugify(human_readable_name: str) -> str:
    site_slug = human_readable_name.strip().lower()
    site_slug = re.sub(r"[^a-z0-9]+", "-", site_slug)
    site_slug = re.sub(r"-+", "-", site_slug).strip("-")
    return site_slug or "site"


def ensure_directory(directory_path: str) -> None:
    os.makedirs(directory_path, exist_ok=True)


def render_svg_to_png(svg_path: str, output_path: str, size_pixels: int) -> None:
    with open(svg_path, "rb") as svg_file_handle:
        svg_bytes = svg_file_handle.read()
    png_bytes = cairosvg.svg2png(
        bytestring=svg_bytes,
        output_width=size_pixels,
        output_height=size_pixels,
    )
    with open(output_path, "wb") as png_file_handle:
        png_file_handle.write(png_bytes)


def generate_multi_resolution_ico(
        source_png_path_for_resampling: str,
        output_ico_path: str,
        ico_sizes: List[Tuple[int, int]],
) -> None:
    base_image = Image.open(source_png_path_for_resampling).convert("RGBA")
    base_image.save(output_ico_path, sizes=ico_sizes)


def write_manifest_json(
        manifest_directory: str,
        site_name_for_manifest: str,
        theme_color: str,
        background_color: str,
        icon_filename_by_size: Dict[int, str],
) -> None:
    manifest_icons: List[Dict[str, str]] = []
    for size_pixels in sorted(icon_filename_by_size.keys()):
        if size_pixels < 192:
            continue
        manifest_icons.append(
            {
                "src": icon_filename_by_size[size_pixels],  # filenames; manifest lives next to icons
                "sizes": f"{size_pixels}x{size_pixels}",
                "type": "image/png",
            }
        )
    manifest_payload = {
        "name": site_name_for_manifest,
        "short_name": site_name_for_manifest,
        "icons": manifest_icons,
        "theme_color": theme_color,
        "background_color": background_color,
        "display": "standalone",
    }
    manifest_path = os.path.join(manifest_directory, "site.webmanifest")
    with open(manifest_path, "w", encoding="utf-8") as manifest_handle:
        json.dump(manifest_payload, manifest_handle, indent=2)


def write_browserconfig_xml(output_dir: str, tile_color: str, url_prefix: str) -> None:
    if not url_prefix.endswith("/"):
        url_prefix += "/"
    xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<browserconfig>
  <msapplication>
    <tile>
      <square150x150logo src="{url_prefix}mstile-150x150.png"/>
      <TileColor>{tile_color}</TileColor>
    </tile>
  </msapplication>
</browserconfig>
"""
    with open(os.path.join(output_dir, "browserconfig.xml"), "w", encoding="utf-8") as xml_handle:
        xml_handle.write(xml_content)


def build_minimal_head_snippet(theme_color: str, url_prefix: str) -> str:
    if not url_prefix.endswith("/"):
        url_prefix += "/"
    return f"""<!-- Favicons (minimal preset) -->
<link rel="icon" type="image/png" sizes="32x32" href="{url_prefix}favicon-32x32.png">
<link rel="apple-touch-icon" sizes="180x180" href="{url_prefix}apple-touch-icon.png">
<link rel="manifest" href="{url_prefix}site.webmanifest">
<link rel="mask-icon" href="{url_prefix}safari-pinned-tab.svg" color="{theme_color}">
<meta name="theme-color" content="{theme_color}">
<meta name="msapplication-TileColor" content="{theme_color}">
<link rel="shortcut icon" href="{url_prefix}favicon.ico">
"""


def build_full_head_snippet(theme_color: str, url_prefix: str) -> str:
    if not url_prefix.endswith("/"):
        url_prefix += "/"
    return f"""<!-- Favicons (full preset) -->
<link rel="icon" type="image/png" sizes="16x16" href="{url_prefix}favicon-16x16.png">
<link rel="icon" type="image/png" sizes="24x24" href="{url_prefix}favicon-24x24.png">
<link rel="icon" type="image/png" sizes="32x32" href="{url_prefix}favicon-32x32.png">
<link rel="icon" type="image/png" sizes="48x48" href="{url_prefix}favicon-48x48.png">
<link rel="icon" type="image/png" sizes="72x72" href="{url_prefix}favicon-72x72.png">
<link rel="icon" type="image/png" sizes="96x96" href="{url_prefix}favicon-96x96.png">
<link rel="icon" type="image/png" sizes="128x128" href="{url_prefix}favicon-128x128.png">
<link rel="icon" type="image/png" sizes="144x144" href="{url_prefix}favicon-144x144.png">
<link rel="icon" type="image/png" sizes="152x152" href="{url_prefix}favicon-152x152.png">
<link rel="icon" type="image/png" sizes="167x167" href="{url_prefix}favicon-167x167.png">
<link rel="icon" type="image/png" sizes="192x192" href="{url_prefix}android-chrome-192x192.png">
<link rel="icon" type="image/png" sizes="256x256" href="{url_prefix}favicon-256x256.png">
<link rel="icon" type="image/png" sizes="384x384" href="{url_prefix}favicon-384x384.png">
<link rel="icon" type="image/png" sizes="512x512" href="{url_prefix}android-chrome-512x512.png">
<link rel="apple-touch-icon" sizes="180x180" href="{url_prefix}apple-touch-icon.png">
<link rel="manifest" href="{url_prefix}site.webmanifest">
<link rel="mask-icon" href="{url_prefix}safari-pinned-tab.svg" color="{theme_color}">
<meta name="theme-color" content="{theme_color}">
<meta name="msapplication-TileColor" content="{theme_color}">
<link rel="shortcut icon" href="{url_prefix}favicon.ico">
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a favicon package from an SVG.")
    parser.add_argument("--svg", required=True, help="Path to source SVG.")
    parser.add_argument("--site-name", default=None, help="Human-readable site name (used for manifest and output slug when provided).")
    parser.add_argument("--assets-path", default="assets/favicons", help="Path under site root where icons will live, e.g., 'assets/favicons'.")
    parser.add_argument("--root-dir", default=".", help="Root directory under which the output folder will be created.")
    parser.add_argument("--theme-color", default="#111111", help="Theme color for browsers (metadata only).")
    parser.add_argument("--background-color", default="#ffffff", help="Background color for manifest (metadata only).")
    parser.add_argument(
        "--preset",
        choices=["minimal", "full"],
        default="minimal",
        help="Which output set to generate. 'minimal' is best-practice; 'full' adds historic sizes."
    )
    parser.add_argument(
        "--windows-tiles",
        action="store_true",
        help="Also generate mstile-150x150.png and browserconfig.xml."
    )
    args = parser.parse_args()

    assets_path_normalized = args.assets_path.strip("/")
    manifest_site_name = args.site_name or "My Site"
    output_folder_name = slugify(args.site_name) if args.site_name else "site"

    absolute_output_dir = os.path.join(os.path.abspath(args.root_dir), output_folder_name, assets_path_normalized)
    ensure_directory(absolute_output_dir)

    url_prefix = "/" + assets_path_normalized + "/"

    core_sizes_for_png = {
        32: "favicon-32x32.png",
        180: "apple-touch-icon.png",
        192: "android-chrome-192x192.png",
        512: "android-chrome-512x512.png",
    }

    full_extra_sizes = [16, 24, 48, 72, 96, 128, 144, 152, 167, 256, 384]

    generated_png_filenames_by_size: Dict[int, str] = {}
    for size_pixels, filename in core_sizes_for_png.items():
        target_path = os.path.join(absolute_output_dir, filename)
        render_svg_to_png(args.svg, target_path, size_pixels)
        generated_png_filenames_by_size[size_pixels] = filename

    if args.preset == "full":
        for size_pixels in full_extra_sizes:
            filename = f"favicon-{size_pixels}x{size_pixels}.png"
            target_path = os.path.join(absolute_output_dir, filename)
            render_svg_to_png(args.svg, target_path, size_pixels)
            generated_png_filenames_by_size[size_pixels] = filename
        sixteen_alias = os.path.join(absolute_output_dir, "favicon-16x16.png")
        if not os.path.exists(sixteen_alias):
            render_svg_to_png(args.svg, sixteen_alias, 16)
            generated_png_filenames_by_size[16] = "favicon-16x16.png"

    shutil.copyfile(args.svg, os.path.join(absolute_output_dir, "safari-pinned-tab.svg"))

    ico_source_png_path = os.path.join(absolute_output_dir, "android-chrome-512x512.png")
    if not os.path.exists(ico_source_png_path):
        ico_source_png_path = os.path.join(absolute_output_dir, "android-chrome-192x192.png")
    ico_output_path = os.path.join(absolute_output_dir, "favicon.ico")
    generate_multi_resolution_ico(ico_source_png_path, ico_output_path, [(16, 16), (32, 32), (48, 48), (64, 64)])

    if args.windows_tiles:
        mstile_target_path = os.path.join(absolute_output_dir, "mstile-150x150.png")
        render_svg_to_png(args.svg, mstile_target_path, 150)
        write_browserconfig_xml(absolute_output_dir, args.theme_color, url_prefix)

    generated_png_filenames_by_size.setdefault(192, "android-chrome-192x192.png")
    generated_png_filenames_by_size.setdefault(512, "android-chrome-512x512.png")
    write_manifest_json(
        manifest_directory=absolute_output_dir,
        site_name_for_manifest=manifest_site_name,
        theme_color=args.theme_color,
        background_color=args.background_color,
        icon_filename_by_size=generated_png_filenames_by_size,
    )

    head_snippet = (
        build_full_head_snippet(args.theme_color, url_prefix)
        if args.preset == "full"
        else build_minimal_head_snippet(args.theme_color, url_prefix)
    )
    with open(os.path.join(absolute_output_dir, "HEAD-snippet.html"), "w", encoding="utf-8") as snippet_handle:
        snippet_handle.write(head_snippet)

    print(f"Output directory: {absolute_output_dir}")
    print("Add this to your <head>:\n")
    print(head_snippet)


if __name__ == "__main__":
    main()
