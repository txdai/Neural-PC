import cv2
import numpy as np
import os
import argparse
from scipy.ndimage import maximum_filter
from scipy.ndimage import label


def find_local_maxima(image, threshold=0.6, min_distance=100):
    """Find local maxima in the template matching result."""
    # Apply maximum filter to find local peaks
    data_max = maximum_filter(image, size=min_distance)
    maxima = image == data_max
    maxima[image < threshold] = False

    # Label connected components
    labeled, num_objects = label(maxima)
    slices = []
    centers = []

    # Find center of mass of each maximum
    for i in range(1, num_objects + 1):
        y, x = np.where(labeled == i)
        if len(x) > 0:  # if the component is not empty
            center_x = int(np.mean(x))
            center_y = int(np.mean(y))
            centers.append((center_x, center_y))
            slices.append((slice(min(y), max(y) + 1), slice(min(x), max(x) + 1)))

    return centers, slices


def find_crosses_template_matching(image, template, debug=False):
    """
    Find crosses using template matching with rotation.
    """
    best_centers = None
    best_score = 0
    best_angle = 0
    best_result = None
    th, tw = template.shape
    center_offset_y, center_offset_x = th // 2, tw // 2

    # Try different rotations of the template
    for angle in range(-10, 11, 2):  # Try -10 to +10 degrees
        # Rotate template
        M = cv2.getRotationMatrix2D((tw / 2, th / 2), angle, 1)
        rotated_template = cv2.warpAffine(template, M, (tw, th))

        # Perform template matching
        result = cv2.matchTemplate(image, rotated_template, cv2.TM_CCOEFF_NORMED)

        # Find local maxima
        corners, _ = find_local_maxima(result, threshold=0.6, min_distance=100)

        # Convert corner coordinates to center coordinates
        centers = [(x + center_offset_x, y + center_offset_y) for x, y in corners]

        # Calculate score as sum of top 4 correlation values
        if len(centers) >= 4:
            values = [result[c[1] - center_offset_y, c[0] - center_offset_x] for c in centers]
            values.sort(reverse=True)
            score = sum(values[:4])

            if score > best_score:
                best_score = score
                best_centers = centers
                best_angle = angle
                best_result = result

    if debug:
        # Visualize template matching result
        debug_result = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        debug_correlation = cv2.normalize(best_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite("debug_2_correlation.png", debug_correlation)

        # Draw all detected centers and template boxes
        for x, y in best_centers:
            # Draw center point
            cv2.circle(debug_result, (int(x), int(y)), 20, (0, 255, 0), -1)
            # Draw template box
            cv2.rectangle(
                debug_result,
                (int(x - center_offset_x), int(y - center_offset_y)),
                (int(x + center_offset_x), int(y + center_offset_y)),
                (0, 0, 255),
                10,
            )
            # Draw score
            score = best_result[int(y - center_offset_y), int(x - center_offset_x)]
            cv2.putText(debug_result, f"score: {score:.2f}", (int(x + 10), int(y + 10)), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
        cv2.imwrite("debug_3_detections.png", debug_result)

    if len(best_centers) < 4:
        raise ValueError(f"Expected 4 crosses, found {len(best_centers)}")

    # Take the 4 strongest matches
    centers_with_scores = [(c[0], c[1], best_result[int(c[1] - center_offset_y), int(c[0] - center_offset_x)]) for c in best_centers]
    centers_with_scores.sort(key=lambda x: x[2], reverse=True)
    best_centers = [(x, y) for x, y, _ in centers_with_scores[:4]]

    return best_centers, best_angle


def process_image(image_path, template_path, width=3000, height=3000, debug=False):
    # Read images
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if debug:
        cv2.imwrite("debug_1_original.png", img)
        cv2.imwrite("debug_1b_template.png", template)

    # Find crosses using template matching
    crosses, angle = find_crosses_template_matching(img, template, debug)

    # Sort crosses (top-left, top-right, bottom-right, bottom-left)
    # First sort by y-coordinate (with some tolerance for slight misalignment)
    y_threshold = 100  # pixels
    crosses_sorted = sorted(crosses, key=lambda p: p[1])
    top_crosses = sorted(crosses_sorted[:2], key=lambda p: p[0])
    bottom_crosses = sorted(crosses_sorted[2:], key=lambda p: p[0])

    top_left, top_right = top_crosses
    bottom_left, bottom_right = bottom_crosses

    if debug:
        # Draw crosses on debug image
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        crosses_ordered = [top_left, top_right, bottom_right, bottom_left]
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]

        for i, (x, y) in enumerate(crosses_ordered):
            cv2.drawMarker(debug_img, (int(x), int(y)), colors[i], cv2.MARKER_CROSS, 30, 3)
            cv2.putText(debug_img, str(i), (int(x + 15), int(y + 15)), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[i], 2)
        cv2.imwrite("debug_4_ordered_crosses.png", debug_img)

    # Define source and destination points for perspective transform
    src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst_pts = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    # Perform perspective transform
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(img, matrix, (width, height))

    if debug:
        cv2.imwrite("debug_5_final_result.png", result)

    return result


def main():
    parser = argparse.ArgumentParser(description="Process SEM images using SIFT-based calibration.")
    parser.add_argument(
        "--input_folder",
        default="../data/binary/",
        help="Path to the folder containing TIFF images",
    )
    parser.add_argument(
        "--output_folder",
        default="../data/corrected/",
        help="Path to the folder to save calibrated images",
    )
    parser.add_argument(
        "--template",
        default="../data/marker.tiff",
        help="Path to the template cross image",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for filename in os.listdir(args.input_folder):
        if filename.lower().endswith((".tif", ".tiff")):
            input_path = os.path.join(args.input_folder, filename)
            output_path = os.path.join(args.output_folder, f"{filename}")

            try:
                calibrated_image = process_image(input_path, args.template, debug=args.debug)
                cv2.imwrite(output_path, calibrated_image)
                print(f"Processed and saved: {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    main()
