#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "opencv-python-headless>=4.9",
#   "numpy>=1.26"
# ]
# ///
import argparse
import sys

import cv2
import numpy as np


def extract_filled_silhouette(
        input_image_path: str,
        output_mask_path: str,
        threshold_value: int = 200,
        closing_kernel_size: int = 15
) -> np.ndarray:
    """
    Reads an image, converts it to grayscale, thresholds to pick up
    all the dark pixels (ink/shading), applies morphological closing
    to fill small gaps, then finds and fills the largest contour.

    Args:
        input_image_path (str):
            Path to the input image (e.g., a JPG/PNG of the engraved portrait).
        output_mask_path (str):
            Path where the final silhouette mask will be saved (PNG).
        threshold_value (int, optional):
            Grayscale cutoff in [0..255]. Pixels darker than this
            become foreground. Default is 200.
        closing_kernel_size (int, optional):
            Size of the square kernel for morphological closing.
            Larger values fill bigger holes but sacrifice fine detail.
            Default is 15.

    Returns:
        np.ndarray:
            A 2D uint8 array matching the input’s dimensions. Pixels
            inside the filled silhouette are 255 (white), background is 0.
    """
    # 1) Load and convert to grayscale
    original_image = cv2.imread(input_image_path)
    if original_image is None:
        raise FileNotFoundError(f"Could not read '{input_image_path}'")
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 2) Threshold to pick up “dark” pixels (ink/shading) as white (255),
    #    and background as black (0). Using THRESH_BINARY_INV flips it.
    _, thresholded = cv2.threshold(
        grayscale_image,
        threshold_value,
        255,
        cv2.THRESH_BINARY_INV
    )

    # 3) Morphological closing to fill small holes/gaps inside the object
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (closing_kernel_size, closing_kernel_size)
    )
    closed_image = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    # 4) Find external contours in the closed image
    contours, _ = cv2.findContours(
        closed_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError(
            "No contours detected. "
            "Try lowering threshold_value or increasing closing_kernel_size."
        )

    # 5) Select the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # 6) Create a blank mask and draw/fill the largest contour
    silhouette_mask = np.zeros_like(grayscale_image)
    cv2.drawContours(
        silhouette_mask,
        [largest_contour],
        contourIdx=-1,
        color=255,
        thickness=cv2.FILLED
    )

    # 7) Save the silhouette mask as a PNG
    cv2.imwrite(output_mask_path, silhouette_mask)
    return silhouette_mask


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract a filled silhouette (binary mask) from an input image, "
            "assuming a single object on a relatively plain background."
        )
    )
    parser.add_argument(
        "--input-file", "-i",
        required=True,
        help="Path to the input image (e.g., JPG or PNG)."
    )
    parser.add_argument(
        "--output-file", "-o",
        required=True,
        help="Path where the output silhouette mask (PNG) will be saved."
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=200,
        help=(
            "Grayscale cutoff in [0..255]. Pixels darker than this become "
            "foreground (white) in the intermediate binary image. Default: 200."
        )
    )
    parser.add_argument(
        "--kernel-size", "-k",
        type=int,
        default=15,
        help=(
            "Size of the square kernel for morphological closing. "
            "Larger values fill bigger internal gaps. Default: 15."
        )
    )

    args = parser.parse_args()

    try:
        extract_filled_silhouette(
            input_image_path=args.input_file,
            output_mask_path=args.output_file,
            threshold_value=args.threshold,
            closing_kernel_size=args.kernel_size
        )
        print(f"Filled silhouette saved to: {args.output_file}")
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
