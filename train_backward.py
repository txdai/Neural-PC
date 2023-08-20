import os
import random
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import SEMBackwardDataset
from backward_model import ConvSDF
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def criterion(output_sdf, target_sdf, output_img, target_img):
    # SDF Loss
    sdf_loss = nn.MSELoss()(output_sdf, target_sdf)

    # Image Loss, pixel-wise cross entropy
    img_loss = nn.BCELoss()(output_img, target_img)

    # Total Loss
    total_loss = sdf_loss + img_loss

    return total_loss


def train(model, dataloader, criterion, optimizer, device, epoch, num_epochs, writer, iterations_to_log=100):
    model.train()
    running_loss = 0.0
    iteration_loss = 0.0
    iterations_per_epoch = len(dataloader)
    iteration = 0
    for input_img, target_img, input_sdf, target_sdf in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        input_img = input_img.to(device)
        target_img = target_img.to(device)
        input_sdf = input_sdf.to(device)
        target_sdf = target_sdf.to(device)

        optimizer.zero_grad()
        output_sdf = model.forward_conv(input_sdf)
        output_img = model.sdf2img(output_sdf)
        loss = criterion(output_sdf, target_sdf, output_img, target_img)
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
        for input_img, target_img, input_sdf, target_sdf in tqdm(dataloader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            input_sdf = input_sdf.to(device)
            target_sdf = target_sdf.to(device)

            output_sdf = model.forward_conv(input_sdf)
            output_img = model.sdf2img(output_sdf)
            loss = criterion(output_sdf, target_sdf, output_img, target_img)

            running_loss += loss.item()

        writer.add_scalar("Loss/val", running_loss / len(dataloader), epoch)

        # Visualization of validation image
        input_img, target_img, input_sdf, target_sdf = next(iter(dataloader))
        input_img = input_img[0].squeeze().cpu()
        input_sdf = input_sdf.to(device)
        output_sdf = model.forward_conv(input_sdf)
        output_img = model.sdf2img(output_sdf)

        # Select the first image from the batch
        target_img = target_img[0].squeeze().cpu()
        output_img = output_img[0].squeeze().cpu()

        # Plot the target and output images
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        ax1.imshow(input_img.numpy(), cmap="gray")
        ax1.set_title("Input Image")
        ax2.imshow(target_img.numpy(), cmap="gray")
        ax2.set_title("Target Image")
        ax3.imshow(output_img.numpy(), cmap="gray")
        ax3.set_title("Output Image")
        ax4.imshow((torch.abs(output_img - target_img)).numpy(), cmap="gray")
        ax4.set_title("Absolute Difference")
        plt.tight_layout()

        # Add figure to TensorBoard
        writer.add_figure("Validation Image Results", fig, epoch)

    return running_loss / len(dataloader)


def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for input_img, target_img, input_sdf, target_sdf in tqdm(dataloader, desc=f"Testing"):
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            input_sdf = input_sdf.to(device)
            target_sdf = target_sdf.to(device)

            output_sdf = model.forward_conv(input_sdf)
            output_img = model.sdf2img(output_sdf)
            loss = criterion(output_sdf, target_sdf, output_img, target_img)

            running_loss += loss.item()

    return running_loss / len(dataloader)


def run_backward(
    path_input,
    path_target,
    device,
    input_size,
    log_dir,
    num_epoch=30,
    writer=None,
):
    batch_size = 8
    image_list = [f for f in os.listdir(path_input) if f.endswith(".png")]
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

    model = ConvSDF(input_size=input_size)
    model.to(device)

    # Training loop with fixed ae
    best_loss = float("inf")
    iterations_to_log = 1  # Save training loss every 10 iterations
    model.freeze_encoder_decoder()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    print("Training SDF")

    for epoch in range(num_epoch):
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch, num_epoch, writer, iterations_to_log)
        val_loss = validate(model, val_loader, criterion, epoch, num_epoch, writer, device)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f"{log_dir}/model_fixed_{epoch}.pth")
            best_model = model
            best_model_name = f"model_fixed_{epoch}.pth"

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
