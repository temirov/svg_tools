# SVG Tools

A collection of utilities for working with images and SVGs. Each script is a self-contained executable
with a `uv` shebang and inline [PEP 723](https://peps.python.org/pep-0723/) metadata.

---

## Requirements

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/) (installed and available in `$PATH`)

Each script declares its dependencies in a `# /// script` block. When you run the script,
`uv` will automatically install the correct dependencies in an isolated environment.

---

## Usage Pattern

1. Make the script executable:

```shell
   chmod +x script.py
```

2. Run it directly:

```shell
./script.py [arguments...]
```

There is no need to call `uv run` explicitly — the shebang handles it.

---

## Scripts

### `generate_color_gradient.py`

Generate a left-to-right gradient based on a single named color (darker shade → lighter shade)
and draw a centered rectangle of a specified color and ratio.

**Usage:**

```shell
./generate_color_gradient.py \
    --gradient-color <COLOR_NAME_OR_HEX> \
    --center-color <COLOR_NAME_OR_HEX> \
    --width <PIXELS> \
    --height <PIXELS> \
    --rectangle-ratio <FLOAT> \
    --output-file <PATH>
```

Example:

```shell
./generate_color_gradient.py \
    --gradient-color turquoise \
    --center-color gold \
    --width 1600 \
    --height 900 \
    --rectangle-ratio 0.5 \
    --output-file turquoise_golden_16x9.png
```

---

### `image_to_data_uri.py`

Center-crop an image to a square, resize it to 32×32 or 64×64, and output a Base64-encoded PNG data URI.

**Usage:**

```shell
./image_to_data_uri.py <input_image> --size <32|64>
```

Prints the data URI to stdout.

---

### `image_to_silhouette.py`

Extract a filled silhouette (binary mask) from an input image (assumes a single dark object on a plain background).

**Usage:**

```shell
./image_to_silhouette.py \
    --input-file <input_image> \
    --output-file <output_mask.png> \
    [--threshold <0–255>] \
    [--kernel-size <integer>]
```

---

### `image_to_svg.py`

Convert a raster image (JPG/PNG/HEIF, etc.) to SVG using one of two algorithms:

* `contours` — edge detection + contour tracing → **stroked** paths (good for sketches/outlines).
* `flat` — CIELAB k-means color quantization → **filled** color regions with optional outline (good for logos/flat art).

HEIF/HEIC is supported via `pillow-heif`.

**Usage:**

```shell
./image_to_svg.py <input_image> \
  [--algorithm {contours,flat}] \
  [--output_suffix <SUFFIX>] \
  [--blur_kernel <ODD_INT>] \
  [--alpha_threshold <0-255>] \
  \
  # Contours mode
  [--edge_low <INT>] [--edge_high <INT>] \
  [--contour_min_area <FLOAT>] \
  [--contour_epsilon_factor <FLOAT>] \
  [--stroke_color <COLOR>] [--stroke_width <FLOAT>] \
  \
  # Flat mode
  [--color_count <INT>] [--region_min_area <FLOAT>] \
  [--region_epsilon_pixels <FLOAT>] [--palette_seed <INT>] \
  [--include_outline] \
  [--outline_low <INT>] [--outline_high <INT>] \
  [--outline_min_area <FLOAT>] \
  [--outline_width <FLOAT>] [--outline_color <COLOR>]
```

**Quick start:**

```shell
# 1) Outline-only (contours, default algorithm)
./image_to_svg.py assets/photo.jpg \
  --blur_kernel 7 --edge_low 100 --edge_high 250 \
  --contour_min_area 50 --contour_epsilon_factor 0.005 \
  --stroke_color black --stroke_width 1.0

# 2) Flat color regions with an outline
./image_to_svg.py assets/logo.png --algorithm flat \
  --color_count 6 --region_min_area 64 --region_epsilon_pixels 0.75 \
  --include_outline --outline_low 80 --outline_high 200 \
  --outline_min_area 48 --outline_width 1.5 --outline_color black
```

**Notes**

* `--blur_kernel` is made odd automatically; `0/1` disables blur.
* `--alpha_threshold` masks out near-transparent pixels for both algorithms.
* In `flat` mode, background is auto-detected from the image border and dropped from the palette.

**Parameter reference**

*Common*

| Flag                | Meaning                                     |
|---------------------|---------------------------------------------|
| `--algorithm`       | `contours` (default) or `flat`.             |
| `--output_suffix`   | Suffix for output SVG filename.             |
| `--blur_kernel`     | Gaussian blur kernel (odd; `0/1` disables). |
| `--alpha_threshold` | Visible-pixel alpha cutoff (0–255).         |

*Contours mode*

| Flag                               | Meaning                                         |
|------------------------------------|-------------------------------------------------|
| `--edge_low`, `--edge_high`        | Canny thresholds.                               |
| `--contour_min_area`               | Minimum area to keep a contour.                 |
| `--contour_epsilon_factor`         | `ε = factor × perimeter` (poly simplification). |
| `--stroke_color`, `--stroke_width` | Stroke styling for contour paths.               |

*Flat mode*

| Flag                                 | Meaning                                        |
|--------------------------------------|------------------------------------------------|
| `--color_count`                      | K for LAB k-means (background auto-dropped).   |
| `--region_min_area`                  | Minimum area for filled regions.               |
| `--region_epsilon_pixels`            | Absolute ε (pixels) for region simplification. |
| `--palette_seed`                     | Random seed for k-means init.                  |
| `--include_outline`                  | Add outline layer over filled regions.         |
| `--outline_low`, `--outline_high`    | Canny thresholds for the outline.              |
| `--outline_min_area`                 | Minimum area for outline contours.             |
| `--outline_width`, `--outline_color` | Outline stroke styling.                        |

---

### `text_to_svg.py`

Render any text into a valid single-path SVG file.

**Usage:**

```shell
./text_to_svg.py "<Your Text Here>" \
    --font <path_to_font.ttf> \
    --out <output_filename.svg> \
    [--size <units_per_em>]
```

Example:

```shell
./text_to_svg.py "Marco Polo Research Lab" \
    --font assets/GreatVibes-Regular.ttf \
    --out title.svg
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
