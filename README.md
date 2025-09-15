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
   ```bash
   chmod +x script.py
````

2. Run it directly:

   ```bash
   ./script.py [arguments...]
   ```

There is no need to call `uv run` explicitly — the shebang handles it.

---

## Scripts

### `generate_color_gradient.py`

Generate a left-to-right gradient based on a single named color (darker shade → lighter shade)
and draw a centered rectangle of a specified color and ratio.

**Usage:**

```bash
./generate_color_gradient.py \
    --gradient-color <COLOR_NAME_OR_HEX> \
    --center-color <COLOR_NAME_OR_HEX> \
    --width <PIXELS> \
    --height <PIXELS> \
    --rectangle-ratio <FLOAT> \
    --output-file <PATH>
```

Example:

```bash
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

```bash
./image_to_data_uri.py <input_image> --size <32|64>
```

Prints the data URI to stdout.

---

### `image_to_silhouette.py`

Extract a filled silhouette (binary mask) from an input image (assumes a single dark object on a plain background).

**Usage:**

```bash
./image_to_silhouette.py \
    --input-file <input_image> \
    --output-file <output_mask.png> \
    [--threshold <0–255>] \
    [--kernel-size <integer>]
```

---

### `image_to_svg.py`

Convert a raster image (JPG, PNG, HEIC, etc.) into an SVG vector file by detecting contours.

**Usage:**

```bash
./image_to_svg.py <input_image> \
    [--output_suffix <SUFFIX>] \
    [--blur <ODD_INTEGER>] \
    [--canny1 <INT>] \
    [--canny2 <INT>] \
    [--min_area <FLOAT>] \
    [--epsilon <FLOAT>] \
    [--stroke_color <COLOR>] \
    [--stroke_width <FLOAT>]
```

Example:

```bash
./image_to_svg.py assets/sample_photo.jpg \
    --output_suffix _outline.svg \
    --blur 5 \
    --canny1 50 \
    --canny2 150 \
    --min_area 100 \
    --epsilon 0.01 \
    --stroke_color black \
    --stroke_width 2
```

---

### `text_to_svg.py`

Render any text into a valid single-path SVG file.

**Usage:**

```bash
./text_to_svg.py "<Your Text Here>" \
    --font <path_to_font.ttf> \
    --out <output_filename.svg> \
    [--size <units_per_em>]
```

Example:

```bash
./text_to_svg.py "Marco Polo Research Lab" \
    --font assets/GreatVibes-Regular.ttf \
    --out title.svg
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
