#!/usr/bin/env python3

import argparse
from PIL import Image, ImageDraw, ImageColor
import numpy as np

def generate_dithered_gradient(left_color_rgb, right_color_rgb, width, height):
    """
    Generate a left-to-right 8-bit gradient between two RGB colors,
    then apply 1-pixel-scale dithering to hide banding.
    """
    # Create a float32 array so we can add sub-1.0 noise before converting to uint8
    gradient_array = np.zeros((height, width, 3), dtype=np.float32)
    left_color = np.array(left_color_rgb, dtype=np.float32)
    right_color = np.array(right_color_rgb, dtype=np.float32)

    # Linear interpolate per column
    for x in range(width):
        alpha = x / (width - 1)
        interpolated = (1 - alpha) * left_color + alpha * right_color
        gradient_array[:, x, :] = interpolated

    # Add uniform noise in [-0.5, +0.5) to each 8-bit channel to dither
    noise = (np.random.rand(height, width, 3).astype(np.float32) - 0.5)
    gradient_array += noise

    # Clamp to [0,255] and convert to uint8
    np.clip(gradient_array, 0, 255, out=gradient_array)
    final_array = gradient_array.astype(np.uint8)

    return Image.fromarray(final_array)

def add_center_rectangle(image, rectangle_ratio, rectangle_color_rgb):
    """Draw a centered rectangle of given ratio and RGB fill onto the image."""
    width, height = image.size
    rect_width = int(width * rectangle_ratio)
    rect_height = int(rect_width * 9 / 16)
    if rect_height > height:
        rect_height = int(height * rectangle_ratio)
        rect_width = int(rect_height * 16 / 9)
    x0 = (width - rect_width) // 2
    y0 = (height - rect_height) // 2
    x1 = x0 + rect_width
    y1 = y0 + rect_height
    draw = ImageDraw.Draw(image)
    draw.rectangle([x0, y0, x1, y1], fill=tuple(rectangle_color_rgb))
    return image

def parse_color_name_to_rgb(color_name):
    """Convert a CSS color name or hex string into an (R, G, B) tuple."""
    return ImageColor.getrgb(color_name)

def compute_darker_and_lighter(rgb_tuple):
    """
    Given an RGB tuple, return a pair of RGB tuples:
    (darker, lighter) by 50% on each end.
    """
    darker = tuple(int(channel * 0.5) for channel in rgb_tuple)
    lighter = tuple(int(channel + (255 - channel) * 0.5) for channel in rgb_tuple)
    return darker, lighter

def main():
    """Parse CLI arguments, generate a dithered gradient, draw a center rectangle, and save."""
    parser = argparse.ArgumentParser(
        description="Generate a dithered gradient from one named color to a lighter shade, then draw a center rectangle."
    )
    parser.add_argument(
        "--gradient-color",
        required=True,
        help="Named color or hex string for the gradient base (e.g. 'red' or '#FF0000')."
    )
    parser.add_argument(
        "--center-color",
        required=True,
        help="Named color or hex string for the rectangle fill (e.g. 'gold' or '#FFD700')."
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Output image width in pixels."
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Output image height in pixels."
    )
    parser.add_argument(
        "--rectangle-ratio",
        type=float,
        default=0.5,
        help="Fraction of image width for rectangle width (height follows 16:9). Default: 0.5."
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path for the generated image (any extension PIL supports, e.g. .jpg, .png)."
    )
    args = parser.parse_args()

    base_rgb = parse_color_name_to_rgb(args.gradient_color)
    left_rgb, right_rgb = compute_darker_and_lighter(base_rgb)

    # Always generate an 8-bit dithered gradient (so that saving to JPEG is possible)
    gradient_image = generate_dithered_gradient(left_rgb, right_rgb, args.width, args.height)

    rectangle_rgb = parse_color_name_to_rgb(args.center_color)
    final_image = add_center_rectangle(gradient_image, args.rectangle_ratio, rectangle_rgb)

    # PIL will pick format by the filename extension (.jpg, .png, etc.)
    final_image.save(args.output_file)

if __name__ == "__main__":
    main()
