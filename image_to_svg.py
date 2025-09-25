#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "opencv-python-headless>=4.9",
#   "numpy>=1.26",
#   "pillow>=10",
#   "pillow-heif>=0.18"
# ]
# ///
"""
image_to_svg.py

Raster-to-SVG converter supporting two algorithms:

1) contours  – edge detection + contour tracing → stroked paths
2) flat      – color quantization in CIELAB → filled color regions (+ optional outline)

Usage examples:

  # Contour outline (default algorithm)
  ./image_to_svg.py input.png --blur_kernel 7 --edge_low 100 --edge_high 250 --contour_min_area 50 --contour_epsilon_factor 0.005 --stroke_color black --stroke_width 1.0

  # Flat regions with optional outline
  ./image_to_svg.py input.png --algorithm flat --color_count 6 --region_min_area 64 --region_epsilon_pixels 0.75 --include_outline --outline_low 80 --outline_high 200 --outline_min_area 48 --outline_width 1.5

Both modes honor HEIF/HEIC via pillow-heif opener registration.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener


register_heif_opener()


@dataclass
class ContourParams:
    gaussian_blur_kernel: int
    canny_low_threshold: int
    canny_high_threshold: int
    minimum_contour_area: float
    approximation_epsilon_factor: float
    stroke_color: str
    stroke_width: float
    alpha_visibility_threshold: int


@dataclass
class FlatParams:
    gaussian_blur_kernel: int
    color_count: int
    region_minimum_area: float
    region_approximation_epsilon_pixels: float
    include_outline_layer: bool
    outline_canny_low: int
    outline_canny_high: int
    outline_minimum_area: float
    outline_stroke_width: float
    outline_stroke_color: str
    alpha_visibility_threshold: int
    palette_random_seed: int


def ensure_odd(value: int) -> int:
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value - 1


def load_image_as_bgra(image_path: str) -> np.ndarray:
    image_pil = Image.open(image_path)
    if image_pil.mode != "RGBA":
        image_pil = image_pil.convert("RGBA")
    image_bgra = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGBA2BGRA)
    return image_bgra


def compute_border_background_color(image_bgra: np.ndarray) -> Tuple[int, int, int]:
    image_height, image_width = image_bgra.shape[:2]
    border_pixels_bgr = np.concatenate(
        [
            image_bgra[0, :, :3],
            image_bgra[image_height - 1, :, :3],
            image_bgra[:, 0, :3],
            image_bgra[:, image_width - 1, :3],
        ],
        axis=0,
    )
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _compactness, _labels, cluster_centers = cv2.kmeans(
        border_pixels_bgr.astype(np.float32), 1, None, termination_criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    dominant_bgr_uint8 = cluster_centers[0].astype(np.uint8)
    return int(dominant_bgr_uint8[0]), int(dominant_bgr_uint8[1]), int(dominant_bgr_uint8[2])


def quantize_in_lab(image_bgra: np.ndarray, parameters: FlatParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      labels_map: HxW int32 of cluster indices (visible pixels assigned, others -1)
      palette_bgr: Kx3 uint8 color centers (in BGR)
    """
    image_bgr = image_bgra[:, :, :3]
    alpha_channel = image_bgra[:, :, 3]

    blur_kernel_size = ensure_odd(parameters.gaussian_blur_kernel)
    if blur_kernel_size > 1:
        image_bgr = cv2.GaussianBlur(image_bgr, (blur_kernel_size, blur_kernel_size), 0)

    visible_mask = alpha_channel >= parameters.alpha_visibility_threshold
    visible_pixels_bgr = image_bgr[visible_mask]
    if visible_pixels_bgr.size == 0:
        labels_map_empty = np.full(image_bgr.shape[:2], fill_value=-1, dtype=np.int32)
        empty_palette = np.zeros((0, 3), dtype=np.uint8)
        return labels_map_empty, empty_palette

    visible_pixels_lab = cv2.cvtColor(
        visible_pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB
    ).reshape(-1, 3)

    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    attempts_count = 3
    initialization_flags = cv2.KMEANS_PP_CENTERS
    np.random.seed(parameters.palette_random_seed)

    cluster_count = max(2, int(parameters.color_count))
    _compactness, labels, centers_lab = cv2.kmeans(
        visible_pixels_lab.astype(np.float32),
        cluster_count,
        None,
        termination_criteria,
        attempts_count,
        initialization_flags,
    )
    centers_lab_uint8 = centers_lab.astype(np.uint8).reshape(-1, 1, 3)
    centers_bgr_uint8 = cv2.cvtColor(centers_lab_uint8, cv2.COLOR_LAB2BGR).reshape(-1, 3).astype(np.uint8)

    labels_map_full = np.full(image_bgr.shape[:2], fill_value=-1, dtype=np.int32)
    labels_map_full[visible_mask] = labels.flatten()
    return labels_map_full, centers_bgr_uint8


def drop_background_cluster(
        labels_map: np.ndarray,
        palette_bgr: np.ndarray,
        image_bgra: np.ndarray,
        background_color_bgr: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    if palette_bgr.shape[0] <= 2:
        return labels_map, palette_bgr

    background_color_vector = np.array(background_color_bgr, dtype=np.float32)
    distances = np.linalg.norm(palette_bgr.astype(np.float32) - background_color_vector[None, :], axis=1)
    background_cluster_index = int(np.argmin(distances))

    mask_background = labels_map == background_cluster_index
    labels_without_bg = labels_map.copy()
    labels_without_bg[mask_background] = -1

    new_palette_bgr = np.delete(palette_bgr, background_cluster_index, axis=0)

    index_remap: Dict[int, int] = {}
    next_index = 0
    for original_index in range(palette_bgr.shape[0]):
        if original_index == background_cluster_index:
            continue
        index_remap[original_index] = next_index
        next_index += 1

    reindexed_labels = labels_without_bg.copy()
    valid_mask = reindexed_labels >= 0
    if np.any(valid_mask):
        vectorized_mapper = np.vectorize(index_remap.get)
        reindexed_labels[valid_mask] = vectorized_mapper(reindexed_labels[valid_mask])
    return reindexed_labels, new_palette_bgr


def trace_contours_for_label(
        labels_map: np.ndarray,
        label_value: int,
        minimum_area: float,
        approximation_epsilon_pixels: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    mask_uint8 = (labels_map == label_value).astype(np.uint8) * 255
    if cv2.countNonZero(mask_uint8) == 0:
        return []

    morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, morphology_kernel, iterations=1)

    contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    results: List[Tuple[np.ndarray, np.ndarray]] = []
    if hierarchy is None:
        return results
    hierarchy_rows = hierarchy[0]

    for index, contour in enumerate(contours):
        area_value = abs(cv2.contourArea(contour))
        if area_value < minimum_area:
            continue
        if approximation_epsilon_pixels > 0.0:
            perimeter_value = max(1.0, cv2.arcLength(contour, True))
            epsilon_value = min(perimeter_value * 0.5, approximation_epsilon_pixels)
            contour = cv2.approxPolyDP(contour, epsilon_value, True)
        results.append((contour, hierarchy_rows[index]))
    return results


def contours_to_svg_path_d(contours_with_hierarchy: List[Tuple[np.ndarray, np.ndarray]]) -> str:
    path_segments: List[str] = []
    for contour, _hierarchy_row in contours_with_hierarchy:
        if contour.shape[0] < 2:
            continue
        points_array = contour.reshape(-1, 2)
        move_segment = f"M {int(points_array[0,0])},{int(points_array[0,1])}"
        line_segments = " ".join([f"L {int(x)},{int(y)}" for x, y in points_array[1:]])
        path_segments.append(f"{move_segment} {line_segments} Z")
    return " ".join(path_segments)


def contours_list_to_paths(contours_list: List[np.ndarray]) -> List[str]:
    path_strings: List[str] = []
    for single_contour in contours_list:
        if single_contour.shape[0] < 2:
            continue
        points_array = single_contour.reshape(-1, 2)
        move_segment = f"M {int(points_array[0,0])},{int(points_array[0,1])}"
        line_segments = " ".join([f"L {int(x)},{int(y)}" for x, y in points_array[1:]])
        path_strings.append(f"{move_segment} {line_segments} Z")
    return path_strings


def bgr_to_hex(color_bgr: np.ndarray) -> str:
    blue, green, red = int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])
    return f"#{red:02x}{green:02x}{blue:02x}"


def build_outline_paths(image_bgra: np.ndarray, low: int, high: int, blur_kernel: int, min_area: float, alpha_threshold: int, epsilon_pixels: float) -> List[str]:
    image_bgr = image_bgra[:, :, :3]
    alpha_channel = image_bgra[:, :, 3]
    visible_mask = alpha_channel >= alpha_threshold

    grayscale_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_kernel_size = ensure_odd(max(3, blur_kernel))
    if blur_kernel_size > 1:
        grayscale_image = cv2.GaussianBlur(grayscale_image, (blur_kernel_size, blur_kernel_size), 0)

    edges_image = cv2.Canny(grayscale_image, low, high)
    edges_image[~visible_mask] = 0

    contours, _hier = cv2.findContours(edges_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    path_strings: List[str] = []
    for contour in contours:
        if abs(cv2.contourArea(contour)) < min_area:
            continue
        perimeter_value = max(1.0, cv2.arcLength(contour, True))
        epsilon_value = min(perimeter_value * 0.5, max(0.0, epsilon_pixels))
        simplified_contour = cv2.approxPolyDP(contour, epsilon_value, True)
        if simplified_contour.shape[0] < 2:
            continue
        points_array = simplified_contour.reshape(-1, 2)
        move_segment = f"M {int(points_array[0,0])},{int(points_array[0,1])}"
        line_segments = " ".join([f"L {int(x)},{int(y)}" for x, y in points_array[1:]])
        path_strings.append(f"{move_segment} {line_segments} Z")
    return path_strings


def write_svg(
        width: int,
        height: int,
        color_layers: List[Tuple[str, str]],
        stroke_layers: List[Tuple[str, str, float]],
        output_svg_path: str,
) -> bool:
    svg_lines: List[str] = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    for fill_hex, path_d in color_layers:
        svg_lines.append(f'<path d="{path_d}" fill="{fill_hex}" stroke="none" fill-rule="evenodd"/>')
    for path_d, stroke_color, stroke_width in stroke_layers:
        svg_lines.append(
            f'<path d="{path_d}" fill="none" stroke="{stroke_color}" stroke-width="{stroke_width}" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    svg_lines.append("</svg>")
    try:
        with open(output_svg_path, "w", encoding="utf-8") as file_pointer:
            file_pointer.write("\n".join(svg_lines))
        return True
    except Exception as error:
        print(f"SVG_WRITE_ERROR: {error}")
        return False


def pipeline_contours(image_bgra: np.ndarray, parameters: ContourParams) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, float]]]:
    image_bgr = image_bgra[:, :, :3]
    alpha_channel = image_bgra[:, :, 3]
    visible_mask = alpha_channel >= parameters.alpha_visibility_threshold

    grayscale_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_kernel_size = ensure_odd(parameters.gaussian_blur_kernel)
    if blur_kernel_size > 1:
        grayscale_image = cv2.GaussianBlur(grayscale_image, (blur_kernel_size, blur_kernel_size), 0)

    edges_image = cv2.Canny(grayscale_image, parameters.canny_low_threshold, parameters.canny_high_threshold)
    edges_image[~visible_mask] = 0

    contours_raw, _hier = cv2.findContours(edges_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    approved_contours: List[np.ndarray] = []
    for contour in contours_raw:
        area_value = cv2.contourArea(contour)
        if area_value < parameters.minimum_contour_area:
            continue
        perimeter_value = cv2.arcLength(contour, True)
        epsilon_value = parameters.approximation_epsilon_factor * perimeter_value
        approximated_contour = cv2.approxPolyDP(contour, epsilon_value, True)
        if len(approximated_contour) >= 2:
            approved_contours.append(approximated_contour)

    stroke_paths = [(path_d, parameters.stroke_color, parameters.stroke_width) for path_d in contours_list_to_paths(approved_contours)]
    return [], stroke_paths


def pipeline_flat(image_bgra: np.ndarray, parameters: FlatParams) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, float]]]:
    image_height, image_width = image_bgra.shape[:2]

    background_bgr = compute_border_background_color(image_bgra)
    labels_map, palette_bgr = quantize_in_lab(image_bgra, parameters)
    labels_map, palette_bgr = drop_background_cluster(labels_map, palette_bgr, image_bgra, background_bgr)

    color_layers: List[Tuple[str, str]] = []
    for color_index, color_bgr in enumerate(palette_bgr):
        contours_with_hierarchy = trace_contours_for_label(
            labels_map=labels_map,
            label_value=color_index,
            minimum_area=parameters.region_minimum_area,
            approximation_epsilon_pixels=parameters.region_approximation_epsilon_pixels,
        )
        if not contours_with_hierarchy:
            continue
        path_d = contours_to_svg_path_d(contours_with_hierarchy)
        fill_hex = bgr_to_hex(color_bgr)
        color_layers.append((fill_hex, path_d))

    stroke_layers: List[Tuple[str, str, float]] = []
    if parameters.include_outline_layer:
        outline_paths = build_outline_paths(
            image_bgra=image_bgra,
            low=parameters.outline_canny_low,
            high=parameters.outline_canny_high,
            blur_kernel=parameters.gaussian_blur_kernel,
            min_area=parameters.outline_minimum_area,
            alpha_threshold=parameters.alpha_visibility_threshold,
            epsilon_pixels=parameters.region_approximation_epsilon_pixels,
        )
        for outline_d in outline_paths:
            stroke_layers.append((outline_d, parameters.outline_stroke_color, parameters.outline_stroke_width))

    if not color_layers and not stroke_layers:
        print("INFO: No vector paths generated by flat algorithm.")
    return color_layers, stroke_layers


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raster images to SVG using either contour tracing or flat-color region vectorization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_image", help="Path to input image (PNG/JPG/HEIF).")
    parser.add_argument("--algorithm", choices=["contours", "flat"], default="contours", help="Vectorization algorithm.")
    parser.add_argument("--output_suffix", default="_vectorized.svg", help="Suffix for output SVG file name.")
    parser.add_argument("--blur_kernel", type=int, default=7, help="Gaussian blur kernel size (odd; 0/1 disables).")
    parser.add_argument("--alpha_threshold", type=int, default=4, help="Alpha threshold to treat pixels as visible (0–255).")

    contour_group = parser.add_argument_group("Contours Mode")
    contour_group.add_argument("--edge_low", type=int, default=100, help="Canny lower threshold.")
    contour_group.add_argument("--edge_high", type=int, default=250, help="Canny upper threshold.")
    contour_group.add_argument("--contour_min_area", type=float, default=50.0, help="Minimum contour area to keep.")
    contour_group.add_argument("--contour_epsilon_factor", type=float, default=0.005, help="Approximation factor × perimeter.")
    contour_group.add_argument("--stroke_color", type=str, default="black", help="Stroke color for contour paths.")
    contour_group.add_argument("--stroke_width", type=float, default=1.0, help="Stroke width for contour paths.")

    flat_group = parser.add_argument_group("Flat Mode")
    flat_group.add_argument("--color_count", type=int, default=6, help="Number of quantized colors (excluding background).")
    flat_group.add_argument("--region_min_area", type=float, default=64.0, help="Minimum area for filled regions.")
    flat_group.add_argument("--region_epsilon_pixels", type=float, default=0.75, help="Absolute simplification epsilon in pixels.")
    flat_group.add_argument("--palette_seed", type=int, default=13, help="Random seed for color k-means.")
    flat_group.add_argument("--include_outline", action="store_true", help="Add outline layer on top of flat regions.")
    flat_group.add_argument("--outline_low", type=int, default=80, help="Canny lower threshold for outline.")
    flat_group.add_argument("--outline_high", type=int, default=200, help="Canny upper threshold for outline.")
    flat_group.add_argument("--outline_min_area", type=float, default=48.0, help="Minimum area for outline contours.")
    flat_group.add_argument("--outline_width", type=float, default=1.5, help="Stroke width for outline layer.")
    flat_group.add_argument("--outline_color", type=str, default="black", help="Stroke color for outline layer.")

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if not os.path.exists(arguments.input_image):
        raise SystemExit(f"Input image not found: {arguments.input_image}")

    base_name, _ext = os.path.splitext(os.path.basename(arguments.input_image))
    output_directory = os.path.dirname(arguments.input_image) or "."
    output_svg_path = os.path.join(output_directory, base_name + arguments.output_suffix)

    image_bgra = load_image_as_bgra(arguments.input_image)
    image_height, image_width = image_bgra.shape[:2]

    if arguments.algorithm == "contours":
        contour_parameters = ContourParams(
            gaussian_blur_kernel=arguments.blur_kernel,
            canny_low_threshold=arguments.edge_low,
            canny_high_threshold=arguments.edge_high,
            minimum_contour_area=arguments.contour_min_area,
            approximation_epsilon_factor=float(arguments.contour_epsilon_factor),
            stroke_color=arguments.stroke_color,
            stroke_width=arguments.stroke_width,
            alpha_visibility_threshold=max(0, min(255, arguments.alpha_threshold)),
        )
        color_layers, stroke_layers = pipeline_contours(image_bgra=image_bgra, parameters=contour_parameters)
    else:
        flat_parameters = FlatParams(
            gaussian_blur_kernel=arguments.blur_kernel,
            color_count=arguments.color_count,
            region_minimum_area=arguments.region_min_area,
            region_approximation_epsilon_pixels=float(arguments.region_epsilon_pixels),
            include_outline_layer=arguments.include_outline,
            outline_canny_low=arguments.outline_low,
            outline_canny_high=arguments.outline_high,
            outline_minimum_area=arguments.outline_min_area,
            outline_stroke_width=arguments.outline_width,
            outline_stroke_color=arguments.outline_color,
            alpha_visibility_threshold=max(0, min(255, arguments.alpha_threshold)),
            palette_random_seed=arguments.palette_seed,
        )
        color_layers, stroke_layers = pipeline_flat(image_bgra=image_bgra, parameters=flat_parameters)

    if not color_layers and not stroke_layers:
        print("FAILURE: no paths produced.")
        raise SystemExit(1)

    success = write_svg(
        width=image_width,
        height=image_height,
        color_layers=color_layers,
        stroke_layers=stroke_layers,
        output_svg_path=output_svg_path,
    )
    if success:
        print(f"SUCCESS: wrote {output_svg_path}")
    else:
        print("FAILURE: could not write SVG.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
