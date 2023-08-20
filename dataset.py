import os
import numpy as np
import random
import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import distance_transform_edt


def binarize(img):
    return (img > 0.5).float()


def compute_signed_distance(binary_img):
    binary_img_np = binary_img.numpy()
    outside = distance_transform_edt(binary_img_np == 0.0)
    inside = distance_transform_edt(binary_img_np == 1.0)
    signed_distance = outside - inside
    return torch.from_numpy(signed_distance)


class SEMForwardDataset(data.Dataset):
    def __init__(
        self,
        path_input,
        path_target,
        image_list,
    ):
        self.path_input = path_input
        self.path_target = path_target
        self.image_list = image_list
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        img_name = self.image_list[index]
        img_input = Image.open(os.path.join(self.path_input, img_name))
        img_target = Image.open(os.path.join(self.path_target, img_name))
        img_input = self.transform(img_input)
        img_target = self.transform(img_target)
        return img_input, img_target

    def __len__(self):
        return len(self.image_list)


class SEMBackwardDatasetOld(data.Dataset):
    def __init__(
        self,
        path_input,
        path_target,
        image_list,
        input_size=512,
        train=True,
    ):
        self.path_input = path_input
        self.path_target = path_target
        self.image_list = image_list
        self.input_size = input_size
        self.train = train
        if train:
            self.crop = self.random_crop
        else:
            self.crop = transforms.CenterCrop(input_size)

    def random_crop(self, img):
        img = TF.rotate(img, self.angle, TF.InterpolationMode.BILINEAR)
        cropped_img = TF.crop(img, *self.crop_coords)
        return cropped_img

    def get_random_crop_coords(self, width, height):
        padding = height // 4 + 1
        left = random.randint(padding, width - self.input_size - padding)
        top = random.randint(padding, height - self.input_size - padding)
        right = self.input_size
        bottom = self.input_size
        self.angle = random.uniform(-180, 180)
        self.crop_coords = (top, left, bottom, right)

    def __getitem__(self, index):
        img_name = self.image_list[index]
        img_input = Image.open(os.path.join(self.path_input, img_name))
        img_target = Image.open(os.path.join(self.path_target, img_name))
        if self.train:
            width, height = img_input.size
            self.get_random_crop_coords(width, height)
        img_input = TF.to_tensor(self.crop(img_input))
        img_target = TF.to_tensor(self.crop(img_target))
        return img_input, img_target

    def __len__(self):
        return len(self.image_list)


class SEMBackwardDataset(data.Dataset):
    def __init__(
        self,
        path_input,
        path_target,
        image_list,
        input_size=512,
        train=True,
    ):
        self.path_input = path_input
        self.path_target = path_target
        self.image_list = image_list
        self.input_size = input_size
        self.train = train
        if train:
            self.crop = self.random_crop
        else:
            self.crop = transforms.CenterCrop(input_size)

    def random_crop(self, img):
        img = TF.rotate(img, self.angle, TF.InterpolationMode.BILINEAR)
        cropped_img = TF.crop(img, *self.crop_coords)
        return cropped_img

    def get_random_crop_coords(self, width, height):
        padding = height // 4 + 1
        left = random.randint(padding, width - self.input_size - padding)
        top = random.randint(padding, height - self.input_size - padding)
        right = self.input_size
        bottom = self.input_size
        self.angle = random.uniform(-180, 180)
        self.crop_coords = (top, left, bottom, right)

    def __getitem__(self, index):
        img_name = self.image_list[index]
        img_input = Image.open(os.path.join(self.path_input, img_name))
        img_target = Image.open(os.path.join(self.path_target, img_name))

        if self.train:
            width, height = img_input.size
            self.get_random_crop_coords(width, height)
        img_input = TF.to_tensor(self.crop(img_input))
        img_target = TF.to_tensor(self.crop(img_target))
        sdf_input = distance_transform_edt(binarize(self.crop(TF.to_tensor(sdf_input))))
        sdf_target = distance_transform_edt(binarize(self.crop(TF.to_tensor(sdf_target))))
        return img_input, img_target, sdf_input, sdf_target

    def __len__(self):
        return len(self.image_list)


class SEMAutoencoderDataset(data.Dataset):
    def __init__(
        self,
        path_input,
        image_list,
        patch_size=64,
        patch_count=10,
        train=True,
    ):
        self.path_input = path_input
        self.image_list = image_list
        self.image_names = []
        for path, img_list in zip(self.path_input, self.image_list):
            self.image_names.extend([os.path.join(path, img_name) for img_name in img_list])

        self.input_size = patch_size
        self.train = train
        if train:
            self.crop = self.random_crop
        else:
            self.crop = transforms.CenterCrop(patch_size)
        self.patch_count = patch_count

    def get_random_patch_coords(self, width, height):
        left = random.randint(0, width - self.input_size)
        top = random.randint(0, height - self.input_size)
        right = self.input_size
        bottom = self.input_size
        self.crop_coords = (top, left, bottom, right)

    def random_crop(self, img):
        cropped_img = TF.crop(img, *self.crop_coords)
        return cropped_img

    def __getitem__(self, index):
        def get_file_extension(full_file_path):
            return os.path.splitext(full_file_path)[1]

        if get_file_extension(self.image_names[index]) == ".npy":
            img_input = np.load(self.image_names[index]) / 255.0
            width = img_input.shape[0]
            height = img_input.shape[1]
            img_input = TF.to_tensor(img_input).to(torch.float)
            self.angle = random.randint(0, 360)
            img_input = TF.rotate(img_input, self.angle, TF.InterpolationMode.BILINEAR)
            patches_input = []
            for _ in range(self.patch_count):
                self.get_random_patch_coords(width, height)
                img_patch = self.crop(img_input)
                patches_input.append(img_patch)
            return patches_input
        else:
            img_input = Image.open(self.image_names[index])
            self.angle = random.randint(0, 360)
            img_input = TF.rotate(img_input, self.angle, TF.InterpolationMode.BILINEAR)
            patches_input = []
            for _ in range(self.patch_count):
                width, height = img_input.size
                self.get_random_patch_coords(width, height)
                img_patch = self.crop(img_input)
                img_patch = TF.to_tensor(img_patch)
                patches_input.append(img_patch)
            return patches_input

    def __len__(self):
        return len(self.image_names)


class SEMTestDataset(data.Dataset):
    def __init__(
        self,
        path_input,
        image_list,
    ):
        self.path_input = path_input
        self.image_list = image_list

    def __getitem__(self, index):
        img_name = self.image_list[index]
        img_input = np.load(os.path.join(self.path_input, img_name))
        img_input = TF.to_tensor(img_input).to(torch.float) / 255.0
        return img_input, img_name

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    path = "./data/dosage1/GDS"
    a = SEMAutoencoderDataset(path_input=[path], image_list=[os.listdir(path)], patch_size=24, patch_count=16, train=True)
    # visualize the dataset
    for i in range(2):
        patches = a[i]
        fig, axs = plt.subplots(1, 16)
        for j, patch in enumerate(patches):
            axs[j].imshow(patch.permute(1, 2, 0))
        plt.show()
