from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
from PIL import Image
import os
import io

log_dir = "./logs/dose1_20230821-175057/"  # Change this to your TensorBoard log directory
save_dir = "../example"  # Change this to where you want to save the images

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Iterate through all event files
for event_file in os.listdir(log_dir):
    event_file_path = os.path.join(log_dir, event_file)

    # Use EventAccumulator to read the event file
    event_acc = EventAccumulator(event_file_path)
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
