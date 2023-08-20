import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image
from dataset import binarize, compute_signed_distance


class RawDataset(data.Dataset):
    def __init__(
        self,
        path_input,
        path_target,
        image_list,
    ):
        self.path_input = path_input
        self.path_target = path_target
        self.image_list = image_list
        self.input_names = []
        self.target_names = []
        for path, img_list in zip(self.path_input, self.image_list):
            self.input_names.extend([os.path.join(path, img_name) for img_name in img_list])
        for path, img_list in zip(self.path_target, self.image_list):
            self.target_names.extend([os.path.join(path, img_name) for img_name in img_list])

    def __getitem__(self, index):
        img_input = Image.open(self.input_names[index])
        img_target = Image.open(self.target_names[index])
        img_input = TF.to_tensor(img_input)
        img_target = TF.to_tensor(img_target)
        structuring_element = torch.ones((1, 1, 41, 41))
        img_input = img_input.unsqueeze_(0)
        img_dilated = torch.nn.functional.conv2d(img_input, structuring_element, padding=20) > 0
        img_target = img_target * img_dilated.squeeze_(0).float()
        return img_target, self.target_names[index]

    def __len__(self):
        return len(self.input_names)


class SDFDataset(data.Dataset):
    def __init__(
        self,
        path_input,
        path_target,
        image_list,
    ):
        self.path_input = path_input
        self.path_target = path_target
        self.image_list = image_list
        self.input_names = []
        self.target_names = []
        for path, img_list in zip(self.path_input, self.image_list):
            self.input_names.extend([os.path.join(path, img_name) for img_name in img_list])
        for path, img_list in zip(self.path_target, self.image_list):
            self.target_names.extend([os.path.join(path, img_name) for img_name in img_list])

    def __getitem__(self, index):
        img_input = Image.open(self.image_names[index])
        img_target = Image.open(self.target_names[index])
        img_input = TF.to_tensor(img_input)
        img_target = TF.to_tensor(img_target)

        # Binarize images
        img_input_binary = binarize(img_input)
        img_target_binary = binarize(img_target)

        # Compute signed distance function
        img_input_sdf = compute_signed_distance(img_input_binary)
        img_target_sdf = compute_signed_distance(img_target_binary)

        return img_input_sdf, img_target_sdf, self.input_names[index], self.target_names[index]

    def __len__(self):
        return len(self.input_names)


path_input = ["./data/dosage1/SEM", "./data/dosage2/SEM", "./data/dosage3/SEM"]
path_target = ["./data/dosage1/GDS", "./data/dosage2/GDS", "./data/dosage3/GDS"]
img_list = [os.listdir(path) for path in path_input]
dataset = SDFDataset(path_input, path_target, img_list)

for img_input, img_target, input_name, target_name in dataset:
    img_input = img_input.numpy()
    img_target = img_target.numpy()
    img_input = np.squeeze(img_input)
    img_target = np.squeeze(img_target)
    input_name = os.path.splitext(input_name)[0]
    target_name = os.path.splitext(target_name)[0]
    np.save(input_name + ".npy", img_input)
    np.save(target_name + ".npy", img_target)
    print(f"Saved {input_name}.npy and {target_name}.npy")
