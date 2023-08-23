from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
from PIL import Image
from test import get_largest_dir_number
import os
import io

for dosage in [1, 2, 3]:
    print(f"Extracting images for dosage {dosage}")
    log_dir = get_largest_dir_number(dosage, "./logs")
    if log_dir == "":
        continue
    log_dir = os.path.join("./logs", log_dir)
    print(f"Log directory: {log_dir}")

    save_dir = "../example"  # Change this to where you want to save the images

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    size_guidance = {
        "images": 0,  # 0 means load all images
    }

    # Iterate through all event files
    for event_file in os.listdir(log_dir):
        event_file_path = os.path.join(log_dir, event_file)

        # Use EventAccumulator to read the event file
        event_acc = EventAccumulator(event_file_path, size_guidance=size_guidance)
        event_acc.Reload()

        # Iterate through all the image tags
        for tag in event_acc.Tags()["images"]:
            images = event_acc.Images(tag)

            # Iterate through the images and save them
            for index, img in enumerate(images):
                image_string = np.frombuffer(img.encoded_image_string, dtype=np.uint8)
                image_data = Image.open(io.BytesIO(image_string))
                image_path = f"{save_dir}/{tag}_{index}.png"  # Change this to the desired path
                image_data.save(image_path)

                print(f"Saved {image_path}")
