import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from typing import Tuple, List, Optional
from tokenizer import BinarySEMTokenizer


class SEMDataset(Dataset):
    """Dataset for loading and randomly cropping SEM images."""

    def __init__(self, corrected_dir: str, design_dir: str, crop_size: int = 2048, transform=None):
        self.crop_size = crop_size
        self.transform = transform

        # Collect image paths
        self.image_paths = []
        for directory in [corrected_dir, design_dir]:
            if directory and os.path.exists(directory):
                self.image_paths.extend([str(p) for p in Path(directory).glob("*.tif*")])

        if not self.image_paths:
            raise ValueError("No images found in the specified directories")

        print(f"Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def random_crop(self, img: torch.Tensor) -> torch.Tensor:
        """Take a random crop from the image."""
        _, h, w = img.shape

        # Calculate valid crop coordinates
        max_x = w - self.crop_size
        max_y = h - self.crop_size

        if max_x < 0 or max_y < 0:
            raise ValueError(f"Image size {(h,w)} too small for crop size {self.crop_size}")

        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        return TF.crop(img, y, x, self.crop_size, self.crop_size)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        img = TF.to_tensor(img)  # Convert to tensor [0,1]

        # Take random crop
        img = self.random_crop(img)
        if self.transform:
            img = self.transform(img)

        return img


def visualize_reconstruction(original: torch.Tensor, reconstructed: torch.Tensor, tokens: List[torch.Tensor], epoch: int, save_dir: str) -> None:
    """Visualize original, reconstructed images and token usage."""
    plt.figure(figsize=(20, 10))

    # Show original and reconstruction
    plt.subplot(2, 3, 1)
    plt.imshow(original[0, 0].cpu().numpy(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(reconstructed[0, 0].cpu().numpy(), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(torch.abs(original[0, 0] - reconstructed[0, 0]).cpu().numpy(), cmap="jet")
    plt.title("Difference")
    plt.axis("off")

    # Show token usage histograms for different scales
    for i, tok in enumerate(tokens[:3]):  # Show first 3 scales
        plt.subplot(2, 3, 4 + i)
        unique, counts = torch.unique(tok[0], return_counts=True)
        plt.bar(unique.cpu().numpy(), counts.cpu().numpy())
        plt.title(f"Scale {i} Token Usage")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"reconstruction_epoch_{epoch}.png"))
    plt.close()


def train_sem_tokenizer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int,
    learning_rate: float = 1e-4,
    save_dir: str = "checkpoints",
    device: str = "cuda",
):
    """Train the SEM tokenizer with codebook maintenance."""

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "viz"), exist_ok=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate / 10)

    # Track best model
    best_loss = float("inf")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, images in enumerate(pbar):
            images = images.to(device)

            # Forward pass
            reconstructed, tokens, loss = model(images)

            # Calculate accuracy
            accuracy = (((reconstructed > 0.5) == (images > 0.5)).float().mean()).item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update statistics
            total_loss += loss.item()
            total_acc += accuracy
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = total_acc / (batch_idx + 1)

            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{avg_acc:.4f}"})

            # Codebook maintenance
            with torch.no_grad():
                # Reset rarely used embeddings
                usage = model.quantizer.ema_vocab_hit.sum(dim=0)
                dead_codes = usage < (usage.mean() * 0.1)  # Reset codes used less than 10% of mean usage
                if dead_codes.any():
                    # Reset dead codes to most used codes + noise
                    most_used = torch.argmax(usage)
                    noise = torch.randn_like(model.quantizer.embedding.weight[0]) * 0.1
                    model.quantizer.embedding.weight.data[dead_codes] = model.quantizer.embedding.weight.data[most_used] + noise

            # Visualize every N batches
            if batch_idx % 100 == 0:
                visualize_reconstruction(images, reconstructed, tokens, epoch, os.path.join(save_dir, "viz"))

        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_acc = 0

            with torch.no_grad():
                for val_images in val_loader:
                    val_images = val_images.to(device)
                    reconstructed, tokens, loss = model(val_images)
                    accuracy = (((reconstructed > 0.5) == (val_images > 0.5)).float().mean()).item()

                    val_loss += loss.item()
                    val_acc += accuracy

                val_loss /= len(val_loader)
                val_acc /= len(val_loader)

                # Save best model
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": best_loss,
                        },
                        os.path.join(save_dir, "best_model.pth"),
                    )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": total_loss / len(train_loader),
            },
            os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"),
        )

        # Update learning rate
        scheduler.step()


def main():
    # Set random seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device("mps")

    # Create dataset
    dataset = SEMDataset(corrected_dir="../data/corrected/", design_dir="../data/design/", crop_size=512)

    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    model = BinarySEMTokenizer(
        image_size=512, vocab_size=4096, embedding_dim=32, num_levels=10, quant_resi=0.5, share_quant_resi=4  # Using crop size
    ).to(device)

    # Train model
    train_sem_tokenizer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=1e-4,
        save_dir="checkpoints",
        device=device,
    )


if __name__ == "__main__":
    main()
