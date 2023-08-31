import os
import random
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.dataset import SEMBackwardDataset
from models.backward_model import ConvAE, Network
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(model, dataloader, criterion, optimizer, device, epoch, num_epochs, writer, iterations_to_log=100):
    model.train()
    running_loss = 0.0
    iteration_loss = 0.0
    iterations_per_epoch = len(dataloader)
    iteration = 0
    for input_img, target_img in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        input_img = input_img.to(device)
        target_img = target_img.to(device)

        optimizer.zero_grad()
        output_img = model.forward_conv(input_img)
        loss = criterion(output_img, target_img)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iteration_loss += loss.item()
        iteration += 1

        if iteration % iterations_to_log == 0:
            writer.add_scalar(
                "Loss/train_iter",
                iteration_loss / iterations_to_log,
                epoch * iterations_per_epoch + iteration,
            )
            iteration_loss = 0.0

    writer.add_scalar("Loss/train", running_loss / len(dataloader), epoch)
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, epoch, num_epochs, writer, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

        writer.add_scalar("Loss/val", running_loss / len(dataloader), epoch)

        # Visualization of validation image
        inputs, targets = next(iter(dataloader))
        input_image = inputs[0].clone().squeeze().cpu()
        inputs = inputs.to(device)
        outputs = model(inputs)

        # Select the first image from the batch
        target_image = targets[0].squeeze().cpu()
        output_image = outputs[0].squeeze().cpu()

        # Plot the target and output images
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        ax1.imshow(input_image.numpy(), cmap="gray")
        ax1.set_title("Input Image")
        ax2.imshow(target_image.numpy(), cmap="gray")
        ax2.set_title("Target Image")
        ax3.imshow(output_image.numpy(), cmap="gray")
        ax3.set_title("Output Image")
        ax4.imshow((torch.abs(output_image - target_image)).numpy(), cmap="gray")
        ax4.set_title("Absolute Difference")
        plt.tight_layout()

        # Add figure to TensorBoard
        writer.add_figure("Validation Image Results", fig, epoch)

    return running_loss / len(dataloader)


def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Testing"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            outputs = torch.clamp(outputs, min=0.0, max=1.0)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

    return running_loss / len(dataloader)


def run_backward(
    path_input,
    path_target,
    device,
    vae,
    latent_size,
    patch_size,
    overlap,
    input_size,
    log_dir,
    num_epoch=30,
    writer=None,
):
    batch_size = 8
    image_list = [f for f in os.listdir(path_input) if f.endswith(".tif")]
    print(f"Number of images: {len(image_list)}")

    # Shuffle the image and sdf lists with the same ordering
    combined = list(zip(image_list, sdf_list))
    random.shuffle(combined)
    image_list, sdf_list = zip(*combined)

    train_list = image_list[: int(len(image_list) * 0.8)]
    val_list = image_list[int(len(image_list) * 0.8) : int(len(image_list) * 0.9)]
    test_list = image_list[int(len(image_list) * 0.9) :]

    train_dataset = SEMBackwardDataset(path_input, path_target, train_list, train=True, input_size=input_size)
    val_dataset = SEMBackwardDataset(path_input, path_target, val_list, train=False, input_size=input_size)
    test_dataset = SEMBackwardDataset(path_input, path_target, test_list, train=False, input_size=input_size)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    conv_model = Network(
        input_size=input_size, large_kernel_sizes=[31, 29, 27, 13], layers=[2, 2, 4, 2], channels=[4, 8, 16, 16], small_kernel=5, device=device
    )
    model = ConvAE(vae_model=vae, conv_model=conv_model, latent_size=latent_size, patch_size=patch_size, overlap=overlap, input_size=input_size).to(
        device
    )

    # Loss function and optimizer
    criterion = nn.MSELoss()

    # Training loop with fixed ae
    best_loss = float("inf")
    iterations_to_log = 1  # Save training loss every 10 iterations
    model.freeze_encoder_decoder()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    print("Training Model")

    for epoch in range(num_epoch):
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch, num_epoch, writer, iterations_to_log)
        val_loss = validate(model, val_loader, criterion, epoch, num_epoch, writer, device)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{log_dir}/model_{epoch}.pth")
            best_model = model
            best_model_name = f"model_{epoch}.pth"

        print(f"Epoch {epoch+1}/{num_epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Testing
    test_loss = test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    # Load the best model
    model.load_state_dict(torch.load(f"{log_dir}/{best_model_name}"))

    # Testing
    test_loss = test(model, test_loader, criterion, device)
    print(f"Best Test Loss: {test_loss:.4f}")

    return best_model, best_model_name
