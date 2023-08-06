import os
import argparse
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import SEMBackwardDataset
from backward_model import ConvAE
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_fixed_ae(model, dataloader, criterion, optimizer, device, epoch, num_epochs, writer, iterations_to_log=100):
    model.train()
    running_loss = 0.0
    iteration_loss = 0.0
    iterations_per_epoch = len(dataloader)
    iteration = 0
    for inputs, targets in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs = model.forward_encoder(inputs)
        targets = model.forward_encoder(targets)

        optimizer.zero_grad()
        outputs = model.forward_conv(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iteration_loss += loss.item()
        iteration += 1

        if iteration % iterations_to_log == 0:
            writer.add_scalar(
                "Loss/fixed_ae_train_iter",
                iteration_loss / iterations_to_log,
                epoch * iterations_per_epoch + iteration,
            )
            iteration_loss = 0.0
            
    writer.add_scalar("Loss/fixed_ae_train", running_loss / len(dataloader), epoch)
    return running_loss / len(dataloader)


def train_finetune_ae(model, dataloader, criterion, optimizer, device, epoch, num_epochs, writer, iterations_to_log=100):
    model.train()
    running_loss = 0.0
    iteration_loss = 0.0
    iterations_per_epoch = len(dataloader)
    iteration = 0
    for inputs, targets in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        iteration_loss += loss.item()
        iteration += 1

        if iteration % iterations_to_log == 0:
            writer.add_scalar(
                "Loss/finetune_ae_train_iter",
                iteration_loss / iterations_to_log,
                epoch * iterations_per_epoch + iteration,
            )
            iteration_loss = 0.0
            
    writer.add_scalar("Loss/finetune_ae_train", running_loss / len(dataloader), epoch)
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
        ax4.imshow((torch.abs(output_image-target_image)).numpy(), cmap="gray")
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
    input_size,
    log_dir,
    num_epochs_fixed=20,
    num_epochs_ae=10,
    latent_size=32,
    encoder=None,
    decoder=None,
    patch_size=24,
    overlap=8,
    writer=None,
):
    batch_size = 8
    image_list = os.listdir(path_input)
    print(f"Number of images: {len(image_list)}")
    train_list = image_list[: int(len(image_list) * 0.8)]
    val_list = image_list[int(len(image_list) * 0.8) : int(len(image_list) * 0.9)]
    test_list = image_list[int(len(image_list) * 0.9) :]

    train_dataset = SEMBackwardDataset(path_input, path_target, train_list, train=True, input_size=input_size)
    val_dataset = SEMBackwardDataset(path_input, path_target, val_list, train=False, input_size=input_size)
    test_dataset = SEMBackwardDataset(path_input, path_target, test_list, train=False, input_size=input_size)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ConvAE(latent_size=latent_size, encoder=encoder, decoder=decoder, patch_size=patch_size, overlap=overlap, input_size=input_size)
    model.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()

    # Training loop with fixed ae
    best_loss = float("inf")
    iterations_to_log = 1  # Save training loss every 10 iterations
    model.freeze_encoder_decoder()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    print("Training without decoder")

    for epoch in range(num_epochs_fixed):
        train_loss = train_fixed_ae(model, train_loader, criterion, optimizer, device, epoch, num_epochs_fixed, writer, iterations_to_log)
        val_loss = validate(model, val_loader, criterion, epoch, num_epochs_fixed, writer, device)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{log_dir}/model_fixed_{epoch}.pth")
            best_model = model
            best_model_name = f"model_fixed_{epoch}.pth"

        print(f"Epoch {epoch+1}/{num_epochs_fixed}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Testing
    test_loss = test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    # Training loop with ae finetuning
    best_loss = float("inf")
    model.freeze_encoder_decoder()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print("Training with decoder")

    for epoch in range(num_epochs_ae):
        train_loss = train_finetune_ae(model, train_loader, criterion, optimizer, device, epoch, num_epochs_ae, writer, iterations_to_log)
        val_loss = validate(model, val_loader, criterion, epoch + num_epochs_fixed, num_epochs_ae + num_epochs_fixed, writer, device)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{log_dir}/model_ae_{epoch}.pth")
            best_model = model
            best_model_name = f"model_ae_{epoch}.pth"

        print(f"Epoch {epoch+1}/{num_epochs_ae}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Load the best model
    model.load_state_dict(torch.load(f"{log_dir}/{best_model_name}"))

    # Testing
    test_loss = test(model, test_loader, criterion, device)
    print(f"Original Test Loss: {test_loss:.4f}")

    return best_model, best_model_name
