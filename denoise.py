import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


class N2VDataset(Dataset):
    def __init__(self, folder_path, patch_size=64, num_pixels_per_patch=64, crop_bottom=250, patches_per_image=128):
        self.patch_size = patch_size
        self.num_pixels_per_patch = num_pixels_per_patch
        self.crop_bottom = crop_bottom
        self.patches_per_image = patches_per_image

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Lambda(lambda x: x.float()),
                transforms.Lambda(lambda x: x / 65535.0 if x.max() > 1.0 else x),
            ]
        )

        self.images = []
        for filename in os.listdir(folder_path):
            if filename.endswith((".tiff", ".tif")):
                img_path = os.path.join(folder_path, filename)
                with Image.open(img_path) as img:
                    img = img.crop((0, 0, img.width, img.height - self.crop_bottom))
                    image = self.transform(img).squeeze(0)
                    self.images.append(image)

        self.generate_patch_indices()

    def generate_patch_indices(self):
        self.patch_indices = []
        for i, img in enumerate(self.images):
            h, w = img.shape
            for _ in range(self.patches_per_image):
                y = torch.randint(0, h - self.patch_size + 1, (1,)).item()
                x = torch.randint(0, w - self.patch_size + 1, (1,)).item()
                self.patch_indices.append((i, y, x))

    def __len__(self):
        return len(self.patch_indices)

    def stratified_random_pixels(self, n_pixels, n_strata):
        strata_size = self.patch_size // n_strata
        pixels = []
        for i in range(n_strata):
            for j in range(n_strata):
                y = i * strata_size + np.random.randint(strata_size)
                x = j * strata_size + np.random.randint(strata_size)
                pixels.append(y * self.patch_size + x)
        return np.random.choice(pixels, n_pixels, replace=False)

    def __getitem__(self, idx):
        img_idx, y, x = self.patch_indices[idx]
        img = self.images[img_idx]
        patch = img[y : y + self.patch_size, x : x + self.patch_size]

        # Stratified sampling to select pixels
        n_strata = int(np.sqrt(self.num_pixels_per_patch))
        pixels = self.stratified_random_pixels(self.num_pixels_per_patch, n_strata)

        # Create mask
        mask = torch.zeros_like(patch).view(-1)
        mask[pixels] = 1
        mask = mask.view(self.patch_size, self.patch_size)

        # Create input patch with blind spots
        input_patch = patch.clone()
        for p in pixels:
            py, px = p // self.patch_size, p % self.patch_size
            neighbors = patch[max(0, py - 1) : min(self.patch_size, py + 2), max(0, px - 1) : min(self.patch_size, px + 2)].flatten()
            input_patch[py, px] = neighbors[torch.randint(0, len(neighbors), (1,))]

        # Create target
        target = patch * mask

        return input_patch.unsqueeze(0), target.unsqueeze(0), mask.unsqueeze(0)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=32):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.conv_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.conv_block(features * 2, features * 4)
        self.upconv1 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(features * 4, features * 2)
        self.upconv2 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(features * 2, features)
        self.final = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d1 = self.decoder1(torch.cat([self.upconv1(b), e2], dim=1))
        d2 = self.decoder2(torch.cat([self.upconv2(d1), e1], dim=1))
        return self.final(d2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )


def train_n2v(folder_path, device="cuda", num_epochs=10, batch_size=32, checkpoint_interval=10):
    dataset = N2VDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if os.path.exists("trained_models/n2v_model.pth"):
        print("Trained model found, skipping training")
        return

    # Load checkpoint if exists
    start_epoch = 0
    if os.path.exists("trained_models/n2v_checkpoint.pth"):
        checkpoint = torch.load("trained_models/n2v_checkpoint.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for masked_batch, target_batch, mask_batch in progress_bar:
            masked_batch, target_batch, mask_batch = masked_batch.to(device), target_batch.to(device), mask_batch.to(device)
            optimizer.zero_grad()
            output = model(masked_batch)
            loss = ((output * mask_batch - target_batch * mask_batch) ** 2).sum() / mask_batch.sum()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

        # Save checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                "trained_models/n2v_checkpoint.pth",
            )
            print(f"Checkpoint saved at epoch {epoch+1}")

    # Save final model
    torch.save(model.state_dict(), "trained_models/n2v_model.pth")


def denoise_image(model, image_path, device="cuda", crop_bottom=250, patch_size=1024, overlap=64):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Lambda(lambda x: x.float()),
            transforms.Lambda(lambda x: x / 65535.0 if x.max() > 1.0 else x),
        ]
    )

    with Image.open(image_path) as img:
        img = img.crop((0, 0, img.width, img.height - crop_bottom))
        image = transform(img).unsqueeze(0)

    _, _, h, w = image.shape
    stride = patch_size - overlap

    # Calculate padding
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    # Apply reflection padding
    image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

    patches = []
    for i in range(0, image_padded.shape[2] - patch_size + 1, stride):
        for j in range(0, image_padded.shape[3] - patch_size + 1, stride):
            patch = image_padded[:, :, i : i + patch_size, j : j + patch_size]
            patches.append(patch)

    denoised_patches = []
    model.eval()
    with torch.no_grad():
        for patch in patches:
            patch = patch.to(device)
            denoised_patch = model(patch)
            denoised_patches.append(denoised_patch.cpu())

    denoised_image = torch.zeros_like(image_padded)
    count = torch.zeros_like(image_padded)

    patch_idx = 0
    for i in range(0, image_padded.shape[2] - patch_size + 1, stride):
        for j in range(0, image_padded.shape[3] - patch_size + 1, stride):
            denoised_image[:, :, i : i + patch_size, j : j + patch_size] += denoised_patches[patch_idx]
            count[:, :, i : i + patch_size, j : j + patch_size] += 1
            patch_idx += 1

    denoised_image /= count

    # Remove padding
    denoised_image = denoised_image[:, :, :h, :w]

    return (denoised_image.squeeze().numpy() * 255).astype(np.uint8)


def denoise_folder(model_path, input_folder, output_folder, device="cuda", crop_bottom=250, patch_size=1024, overlap=64):
    os.makedirs(output_folder, exist_ok=True)
    model = UNet()  # Make sure to import or define UNet
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    file_list = [f for f in os.listdir(input_folder) if f.endswith((".tiff", ".tif"))]
    progress_bar = tqdm(file_list, desc="Denoising images")

    for filename in progress_bar:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"denoised_{filename}")

        denoised_image = denoise_image(model, input_path, device, crop_bottom, patch_size, overlap)
        Image.fromarray(denoised_image).save(output_path)
        progress_bar.set_postfix({"file": filename})


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    # Training
    sem_images_folder = "../data/original/"
    trained_model = train_n2v(sem_images_folder, device=device)

    # Denoising a folder of images
    model_path = "trained_models/n2v_model.pth"
    output_folder = "../data/denoised/"
    denoise_folder(model_path, sem_images_folder, output_folder, device=device)

    print("Denoising complete!")
