import cv2
import numpy as np
from skimage.feature import canny
from skimage.morphology import flood_fill
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import os
import argparse


def is_cross(contour, filled_shape):
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    # Check aspect ratio
    aspect_ratio = w / h
    if not (0.8 < aspect_ratio < 1.2):  # Cross should be roughly square
        return False
    # Check for presence of four extremities
    roi = filled_shape[y : y + h, x : x + w]
    center = (w // 2, h // 2)
    # Check pixels in four directions from center
    directions = [(0, -h // 3), (0, h // 3), (-w // 3, 0), (w // 3, 0)]
    extremities = sum(1 for dx, dy in directions if roi[center[1] + dy, center[0] + dx] > 0)
    return extremities == 4


def process_image(image_path, width=3000, height=3000, debug=False):
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if debug:
        cv2.imwrite("debug_1_original.png", img)

    # Edge detection
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = canny(img)
    if debug:
        cv2.imwrite("debug_2_edges.png", (edges * 255).astype(np.uint8))

    # perform closing to connect edges
    kernel = np.ones((9, 9), np.uint8)
    edges = cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    if debug:
        cv2.imwrite("debug_2_edges_closed.png", (edges * 255).astype(np.uint8))

    # Create a slightly larger image to ensure flood fill reaches the edges
    h, w = edges.shape
    padded = np.pad(edges, pad_width=1, mode="constant", constant_values=0)

    # Flood fill from the border
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded.astype(np.uint8), mask, (0, 0), 1)

    # Remove padding
    filled_outside = mask[2:-2, 2:-2]

    # Invert to get the shapes
    binary = (1 - filled_outside) * 255
    if debug:
        cv2.imwrite("debug_3_binary.png", binary.astype(np.uint8))

    # Find contours
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crosses = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 180 < w < 260 and 160 < h < 240 and is_cross(contour, binary):
            crosses.append((x + w // 2, y + h // 2))

    if len(crosses) != 4:
        raise ValueError(f"Expected 4 crosses, found {len(crosses)}")

    # Sort crosses (top-left, top-right, bottom-right, bottom-left)
    crosses.sort(key=lambda p: p[0] + p[1])
    crosses[1:3] = sorted(crosses[1:3], key=lambda p: p[0] - p[1])

    if debug:
        # Draw crosses on debug image
        debug_img = cv2.cvtColor(binary.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for i, (x, y) in enumerate(crosses):
            cv2.drawMarker(debug_img, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(debug_img, str(i), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite("debug_4_detected_crosses.png", debug_img)

    # Calculate the corners of the output image
    top_left, top_right, bottom_right, bottom_left = crosses[0], crosses[1], crosses[3], crosses[2]

    # Define source and destination points for perspective transform
    src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    # Perform perspective transform
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(img, matrix, (width, height))
    result_binary = cv2.warpPerspective(binary, matrix, (width, height))

    if debug:
        cv2.imwrite("debug_5_final_result.png", result)

    return result, result_binary


def main():
    parser = argparse.ArgumentParser(description="Process SEM images for calibration.")
    parser.add_argument(
        "--input_folder",
        default="../data/denoised/",
        help="Path to the folder containing TIFF images",
    )
    parser.add_argument(
        "--output_folder",
        default="../data/corrected/",
        help="Path to the folder to save calibrated images",
    )
    parser.add_argument("--debug", default=False, help="Enable debug mode")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for filename in os.listdir(args.input_folder):
        if filename.lower().endswith((".tif", ".tiff")):
            input_path = os.path.join(args.input_folder, filename)
            output_path = os.path.join(args.output_folder, f"calibrated_{filename}")

            try:
                calibrated_image, binary_image = process_image(input_path, debug=args.debug)
                cv2.imwrite(output_path, calibrated_image)
                cv2.imwrite(output_path.replace(".tif", "_binary.tif"), binary_image)
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    main()
