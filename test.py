import os
import re
import cv2
import csv
import numpy as np
import torch
import torch.utils.data as data
from dataset import SEMTestDataset, SEMBackwardDatasetOld
from backward_model import ConvSDF
from tqdm import tqdm


def get_largest_model_number(dir_path, model_type):
    max_num = 0
    model_file = ""
    for filename in os.listdir(dir_path):
        if model_type == "backward":
            match = re.match(r"model_(\d+).pth", filename)
        elif model_type == "ae":
            match = re.match(r"ae_model_(\d+).pth", filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                model_file = filename
    return model_file


def get_largest_dir_number(dosage, root_path):
    max_num = 0
    directory = ""
    for filename in os.listdir(root_path):
        match = re.match(r"dose{}_(\d+)-(\d+)".format(dosage), filename)
        if match:
            num = int(match.group(1)) + int(match.group(2))
            if num > max_num:
                max_num = num
                directory = filename
    return directory


def get_largest_ae_number(root_path):
    max_num = 0
    directory = ""
    for filename in os.listdir(root_path):
        match = re.match(r"ae_(\d+)-(\d+)", filename)
        if match:
            num = int(match.group(1)) + int(match.group(2))
            if num > max_num:
                max_num = num
                directory = filename
    return directory


def test(model, dataloader, log_dir, device, input_size):
    model.eval()
    padding = (input_size - 512) // 2

    with torch.no_grad():
        for inputs, filename in tqdm(dataloader, desc="Generating Corrected Samples"):
            inputs = inputs.to(device)
            inputs = torch.nn.functional.pad(inputs, (padding, padding, padding, padding), mode="constant", value=0)
            outputs = model(inputs)
            # outputs = outputs[:, :, padding:-padding, padding:-padding]
            np.save(log_dir + filename[0], outputs.squeeze().detach().cpu().numpy())
            polygons = tensor_to_polygons(outputs.squeeze().detach().cpu().numpy())
            with open(log_dir + filename[0][:-4] + ".csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                for polygon in polygons:
                    if len(polygon) > 2:
                        writer.writerow([coord for point in polygon for coord in point])
    return


def test_loss(model, dataloader, device):
    model.eval()
    running_loss = 0.0
    raw_loss = 0.0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Measuring Loss"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            orig_loss = criterion(inputs, targets)
            outputs = model(inputs)
            mask = outputs > 0.5
            outputs = mask.float().to(device)
            loss = criterion(outputs, targets)

            raw_loss += orig_loss.item()
            running_loss += loss.item()

    return running_loss / len(dataloader), raw_loss / len(dataloader)


def tensor_to_polygons(array, threshold=0.5):
    _, thresh = cv2.threshold(array, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.squeeze().tolist() for contour in contours]
    return polygons


def run_test(dosage, device, input_size):
    root_dir = get_largest_dir_number(dosage, "./logs")
    model_name = get_largest_model_number(f"./logs/{root_dir}", "backward")

    model = ConvSDF(input_size=input_size).to(device)
    model.load_state_dict(torch.load(f"./logs/{root_dir}/{model_name}", map_location=device))

    image_list = os.listdir("./data/data")

    test_dataset = SEMTestDataset(
        "./data/data",
        image_list,
    )

    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    test(model, test_loader, f"./data/corrected_dosage{dosage}/", device, input_size)

    path_input = f"./data/dosage{dosage}/SEM"
    path_target = f"./data/dosage{dosage}/GDS"
    test_list = [f for f in os.listdir(path_input) if f.endswith(".png")]
    test_loss_dataset = SEMBackwardDatasetOld(path_input, path_target, test_list, train=False, input_size=input_size)
    test_loss_loader = data.DataLoader(test_loss_dataset, batch_size=8, shuffle=True, num_workers=0)

    testing_loss, raw_test_loss = test_loss(model, test_loss_loader, device)
    print(f"Binarized Test loss: {testing_loss}, Raw test loss: {raw_test_loss}")


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    run_test(1, device, 512)
