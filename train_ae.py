import os
import torch
import random
import torch.optim as optim
from tqdm import tqdm
import torchvision
from dataset import SEMAutoencoderDataset
from vae_model import VAE, vae_loss
import matplotlib.pyplot as plt


def train(model, dataloader, optimizer, device, writer=None, epoch=0, num_epochs=100):
    model.train()
    running_loss = 0.0

    for patches in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", mininterval=1):
        for patch in patches:
            patch = patch.to(device)
            patch = patch.view(patch.size(0), -1)
            x_reconst, mu, log_var = model(patch)

            loss = vae_loss(patch, x_reconst, mu, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    writer.add_scalar(
        "Loss/train_loss",
        running_loss / len(dataloader),
        epoch,
    )
    return running_loss / len(dataloader)


def validate(model, dataloader, device, writer=None, epoch=0, num_epochs=100, patch_size=16):
    model.eval()
    running_loss = 0.0
    show_image = True

    with torch.no_grad():
        for patches in tqdm(dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", mininterval=1):
            for patch in patches:
                patch = patch.to(device)
                patch = patch.view(patch.size(0), -1)
                x_reconst, mu, log_var = model(patch)

                loss = vae_loss(patch, x_reconst, mu, log_var)

                running_loss += loss.item()

                if show_image:
                    show_image = False

                    fig, axs = plt.subplots(1, 2, figsize=(15, 15))  # Adjust figsize as needed

                    # Input Images
                    grid_img_input = axs[0]
                    grid_img_input.imshow(
                        torchvision.utils.make_grid(patch.view(-1, 1, patch_size, patch_size), nrow=8).cpu().numpy().transpose((1, 2, 0)), cmap="gray"
                    )
                    grid_img_input.axis("off")
                    grid_img_input.set_title("Input Images", fontsize=16)

                    # Reconstructed Images
                    grid_img_reconstr = axs[1]
                    grid_img_reconstr.imshow(
                        torchvision.utils.make_grid(x_reconst.view(-1, 1, patch_size, patch_size), nrow=8).cpu().numpy().transpose((1, 2, 0)), cmap="gray"
                    )
                    grid_img_reconstr.axis("off")
                    grid_img_reconstr.set_title("Reconstructed Images", fontsize=16)

                    plt.tight_layout()
                    writer.add_figure("Input and Reconstruction", fig, epoch)

    writer.add_scalar(
        "Loss/val_loss",
        running_loss / len(dataloader),
        epoch,
    )
    return running_loss / len(dataloader)


def run_ae(paths, patch_size, patch_count, latent_dim, num_epochs, device, writer=None, log_dir="./"):
    batch_size = 64
    img_lists = [os.listdir(path) for path in paths]
    for img_list in img_lists:
        random.shuffle(img_list)
    train_list = [img_list[: int(len(img_list) * 0.8)] for img_list in img_lists]
    valid_list = [img_list[int(len(img_list) * 0.8) :] for img_list in img_lists]

    train_dataset = SEMAutoencoderDataset(path_input=paths, image_list=train_list, patch_size=patch_size, patch_count=patch_count, train=True)
    valid_dataset = SEMAutoencoderDataset(path_input=paths, image_list=valid_list, patch_size=patch_size, patch_count=patch_count, train=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = VAE(input_shape=patch_size * patch_size, latent_dim=latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_loss = float("inf")
    best_model = None
    best_model_name = None
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device, writer=writer, epoch=epoch, num_epochs=num_epochs)
        val_loss = validate(model, valid_loader, device, writer=writer, epoch=epoch, num_epochs=num_epochs, patch_size=patch_size)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{log_dir}/ae_model_{epoch}.pth")
            best_model = model
            best_model_name = f"ae_model_{epoch}.pth"

    return best_model, best_model_name


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    import datetime

    path = "./data/dosage1/GDS"
    root_dir = f"runs/ae_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    patch_size = 24
    latent_dim = 32
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    writer = SummaryWriter(root_dir)
    best_model, best_model_name = run_ae(path, patch_size, 16, latent_dim, 100, device, writer=writer, log_dir=root_dir)
    writer.close()
