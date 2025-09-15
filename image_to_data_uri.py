#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pillow>=10",
#   "pillow-heif>=0.18"
# ]
# ///
"""
Convert a PNG image to a base64-encoded data URI after center-cropping to a square
and resizing to a specified dimension.
"""
import sys
import argparse
import base64
import io
from PIL import Image

def process_image(image_path: str, target_size: int) -> Image.Image:
    """
    Load image, center-crop to square, and resize to target_size x target_size.
    """
    img = Image.open(image_path)
    width, height = img.size
    if width != height:
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        img = img.crop((left, top, left + side, top + side))
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.LANCZOS)
    return img

def image_to_data_uri(img: Image.Image) -> str:
    """
    Encode a PIL Image object as a PNG data URI.
    """
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    b64_data = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b64_data}"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read a PNG, center-crop and resize it, then output a base64 data URI."
    )
    parser.add_argument(
        "input_path",
        help="Path to the input PNG file"
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        choices=[32, 64],
        required=True,
        help="Output image width & height in pixels (32 or 64)"
    )
    args = parser.parse_args()

    img = process_image(args.input_path, args.size)
    uri = image_to_data_uri(img)
    sys.stdout.write(uri)

if __name__ == "__main__":
    main()
