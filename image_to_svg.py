#!/usr/bin/env python3

import argparse
import os

import cv2
import numpy as np
from PIL import Image


# --- SVG Saving Utility ---
def save_contours_to_svg(contours, output_svg_path, width, height,
                         stroke_color="black", stroke_width=1, fill_color="none"):
    if not contours:
        print(f"SVG_SAVE_INFO: No contours were provided to save for '{output_svg_path}'. SVG will not be created.")
        return False

    svg_paths = []
    for contour_idx, contour in enumerate(contours):
        if len(contour) < 2:
            continue

        points_str = "M " + " L ".join([f"{pt[0][0]},{pt[0][1]}" for pt in contour])
        points_str += " Z"
        svg_paths.append(
            f'<path d="{points_str}" stroke="{stroke_color}" stroke-width="{stroke_width}" fill="{fill_color}" />')

    if not svg_paths:
        print(
            f"SVG_SAVE_INFO: No valid path strings generated from contours for '{output_svg_path}'. SVG will not be created.")
        return False

    svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
    svg_content += "\n".join(svg_paths)
    svg_content += "\n</svg>"

    try:
        with open(output_svg_path, 'w') as f:
            f.write(svg_content)
        return True
    except IOError as e:
        print(f"SVG_SAVE_ERROR: IOError saving SVG to '{output_svg_path}': {e}.")
        return False
    except Exception as e:
        print(f"SVG_SAVE_ERROR: Unexpected error saving SVG to '{output_svg_path}': {e}")
        return False


# --- Core Image Processing and SVG Generation Function ---
def convert_image_to_svg(
        image_path,
        output_svg_filename,
        blur_ksize=(7, 7),  # Default from your "ideal" parameters
        canny_threshold1=100,  # Default from your "ideal" parameters
        canny_threshold2=250,  # Default from your "ideal" parameters
        approximation_epsilon_factor=0.005,  # Default from your "ideal" parameters
        min_contour_area=50,  # Default from your "ideal" parameters
        svg_stroke_color="black",
        svg_stroke_width=1,
        svg_fill_color="none"
):
    print(f"PROCESS: Starting image to SVG conversion for '{image_path}'")

    # 1. Load Image
    try:
        if not os.path.exists(image_path):
            print(f"PROCESS_ERROR: Input image file not found at '{image_path}'")
            return False

        pil_img = Image.open(image_path)
        if pil_img.mode == 'P':
            pil_img = pil_img.convert("RGBA")
            img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
        elif pil_img.mode == 'RGBA':
            img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
        elif pil_img.mode == 'RGB':
            img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif pil_img.mode in ['L', '1']:
            img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_GRAY2BGR)
        else:
            print(f"PROCESS_WARNING: Unsupported PIL image mode '{pil_img.mode}'. Attempting direct RGB conversion.")
            img_cv = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

        if img_cv is None:
            print(f"PROCESS_ERROR: OpenCV could not decode image from '{image_path}' (img_cv is None).")
            return False

        original_height, original_width = img_cv.shape[:2]
        print(f"PROCESS: Image loaded. Dimensions: {original_width}x{original_height}, Original Mode: {pil_img.mode}")

    except FileNotFoundError:
        print(f"PROCESS_ERROR: Input image file not found at '{image_path}'.")
        return False
    except Exception as e:
        print(f"PROCESS_ERROR: Could not load or convert image '{image_path}'. Error: {e}")
        return False

    # 2. Preprocessing
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    if blur_ksize[0] > 0 and blur_ksize[1] > 0:
        blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    else:
        blurred = gray

    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    # Canny debug image is removed as per previous request.
    # If you need it back temporarily for this specific image, you can re-add:
    # cv2.imwrite(f"{os.path.splitext(os.path.basename(image_path))[0]}_canny_edges.png", edges)

    if cv2.countNonZero(edges) == 0:
        print("PROCESS_WARNING: Canny image is all black! No edges were detected. No SVG will be generated.")
        print("                 Try adjusting Canny thresholds (lower --canny1, --canny2) or --blur settings.")
        return False

    # 3. Find Contours
    contours_raw, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours_raw:
        print("PROCESS_WARNING: cv2.findContours found no contours. No SVG will be generated.")
        return False

    # 4. Process Contours
    approved_contours_for_svg = []
    for contour in contours_raw:
        if cv2.contourArea(contour) < min_contour_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = approximation_epsilon_factor * perimeter
        approximated_contour = cv2.approxPolyDP(contour, epsilon, True)

        if len(approximated_contour) >= 2:
            approved_contours_for_svg.append(approximated_contour)

    print(f"PROCESS: Retained {len(approved_contours_for_svg)} contours after filtering.")

    # 5. Save SVG file
    if not approved_contours_for_svg:
        print(f"PROCESS_INFO: No approved contours remaining to save to SVG '{output_svg_filename}'.")
        return False

    print(f"PROCESS: Attempting to save {len(approved_contours_for_svg)} contours to SVG: '{output_svg_filename}'")
    svg_saved_successfully = save_contours_to_svg(
        approved_contours_for_svg, output_svg_filename,
        original_width, original_height,
        svg_stroke_color, svg_stroke_width, svg_fill_color
    )

    return svg_saved_successfully


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(
        description="Convert an image (JPG, PNG, etc.) to an SVG vector file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_image", help="Path to the input image file.")

    parser.add_argument("--output_suffix", default="_vectorized.svg",
                        help="Suffix for the output SVG file name (default: '_vectorized.svg').")

    param_group = parser.add_argument_group('Tuning Parameters')
    param_group.add_argument("--blur", type=int, default=7,  # IDEAL default
                             help="Gaussian blur kernel size (positive, odd). 0 or 1 for no blur. (default: 7)")
    param_group.add_argument("--canny1", type=int, default=100,  # IDEAL default
                             help="Canny edge detector: lower threshold (default: 100).")
    param_group.add_argument("--canny2", type=int, default=250,  # IDEAL default
                             help="Canny edge detector: upper threshold (default: 250).")
    param_group.add_argument("--min_area", type=float, default=50.0,  # IDEAL default
                             help="Minimum contour area to keep (default: 50.0).")
    param_group.add_argument("--epsilon", type=float, default=0.005,  # IDEAL default
                             help="Contour approximation factor (default: 0.005).")
    param_group.add_argument("--stroke_color", type=str, default="black", help="SVG path stroke color (default: black)")
    param_group.add_argument("--stroke_width", type=float, default=1.0, help="SVG path stroke width (default: 1.0)")

    args = parser.parse_args()

    print("--- Image to SVG Vectorizer ---")
    if not os.path.exists(args.input_image):
        print(f"CRITICAL_ERROR: Input image '{args.input_image}' not found. Exiting.")
        exit(1)

    base_name, _ = os.path.splitext(os.path.basename(args.input_image))
    output_directory = os.path.dirname(args.input_image) or "."
    svg_file_output = os.path.join(output_directory, base_name + args.output_suffix)

    print(f"Input image:        '{args.input_image}'")
    print(f"Target SVG output:  '{svg_file_output}'")

    print("\nParameters for processing (using defaults or command-line overrides):")
    blur_kernel_actual = (args.blur, args.blur)
    if args.blur <= 1:
        blur_kernel_actual = (0, 0)
    elif args.blur % 2 == 0:  # Make blur kernel odd if it's even and > 1
        blur_kernel_actual = (args.blur - 1, args.blur - 1) if args.blur > 1 else (0, 0)
        if blur_kernel_actual[0] <= 0: blur_kernel_actual = (0, 0)  # ensure it's not 0 or less after adjustment
        print(f"INFO: Blur kernel size was even ({args.blur}), adjusted to be odd: {blur_kernel_actual[0]}.")

    print(f"  Blur Kernel:         {blur_kernel_actual if blur_kernel_actual[0] > 0 else 'No blur'}")
    print(f"  Canny Thresholds:    ({args.canny1}, {args.canny2})")
    print(f"  Min Contour Area:    {args.min_area}")
    print(f"  Approximation Factor: {args.epsilon}")
    print(f"  SVG Stroke Color:    {args.stroke_color}")
    print(f"  SVG Stroke Width:    {args.stroke_width}")
    print("----------------------------------\n")

    svg_was_saved = convert_image_to_svg(
        image_path=args.input_image,
        output_svg_filename=svg_file_output,
        blur_ksize=blur_kernel_actual,  # Pass the validated/adjusted blur kernel
        canny_threshold1=args.canny1,
        canny_threshold2=args.canny2,
        approximation_epsilon_factor=args.epsilon,
        min_contour_area=args.min_area,
        svg_stroke_color=args.stroke_color,
        svg_stroke_width=args.stroke_width,
        svg_fill_color="none"
    )

    print("\n--- Processing Finished ---")
    if svg_was_saved:
        print(f"SUCCESS: SVG file was saved to '{svg_file_output}'")
    else:
        print(f"FAILURE: SVG file was NOT saved for '{svg_file_output}'.")
        print(f"         Please check the log messages above for reasons.")
        print(
            f"         Common issues: incorrect Canny/blur parameters, or all contours filtered by area/approximation.")
    print("---------------------------")


if __name__ == '__main__':
    main()
