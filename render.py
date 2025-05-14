import klayout.lay as lay
import klayout.db as db
import os


def capture_gds_region(gds_path, output_path, x1, y1, x2, y2, width=3000, height=3000, layer_specs=None, background_color="#000000"):
    """
    Capture a screenshot of a specific region in a GDS file.

    Args:
        gds_path (str): Path to GDS file
        output_path (str): Where to save the screenshot
        x1, y1, x2, y2 (float): Coordinates in micrometers
        width (int): Output image width in pixels
        height (int): Output image height in pixels
        layer_specs (list): List of tuples (layer, datatype, color) to render
        background_color (str): Background color in hex format
    """
    lv = lay.LayoutView()
    lv.set_config("background-color", background_color)
    lv.set_config("grid-visible", "false")
    lv.set_config("grid-show-ruler", "false")
    lv.set_config("text-visible", "false")

    lv.load_layout(gds_path, 0)
    lv.clear_layers()

    if layer_specs is None:
        layer_specs = [(0, 0, 0xFFFFFF)]

    for layer, datatype, color in layer_specs:
        lp = lay.LayerProperties()
        lp.source = f"{layer}/{datatype}"
        lp.dither_pattern = 0
        lp.fill_color = color
        lp.frame_color = color
        lv.insert_layer(lv.begin_layers(), lp)

    lv.max_hier()
    lv.timer()
    lv.save_image_with_options(output_path, width, height, 0, 0, 0, db.DBox(x1, y1, x2, y2), False)


def scan_gds_in_grid(gds_path, output_folder, window_size=16, steps=7):
    """
    Scan through GDS file in a grid pattern and save images.

    Args:
        gds_path (str): Path to GDS file
        output_folder (str): Folder to save output images
        window_size (float): Size of each window in micrometers
        steps (int): Number of steps to take in each direction
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Scan through the grid
    for i in range(steps):
        for j in range(steps):
            # Calculate window coordinates
            x1 = -0.3 + i * window_size
            y1 = -0.3 + j * window_size
            x2 = x1 + 8
            y2 = y1 + 8

            # Create output filename
            output_path = os.path.join(output_folder, f"x{i}y{j}.tif")

            # Capture the region
            capture_gds_region(gds_path=gds_path, output_path=output_path, x1=x1, y1=y1, x2=x2, y2=y2, width=3000, height=3000)
            print(f"Captured region at x={i}, y={j}")


if __name__ == "__main__":
    # Scan the entire GDS file
    scan_gds_in_grid(
        gds_path="../data/design_thickness_0_current_2nA_dosage_1.gds",
        output_folder="../data/design/",
        window_size=16,  # 16 micrometer windows
        steps=7,  # 7 steps in each direction
    )
