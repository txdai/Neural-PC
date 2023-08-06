import os
import csv
from tqdm import tqdm
import torch
from fft_conv_pytorch import fft_conv
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from dataset import SEMForwardDataset
import torch.optim as optim

path_input = "./data/processed_GDS"
path_target = "./data/processed_SEM"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")


class GaussianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(GaussianConv2d, self).__init__()

        assert kernel_size % 2 == 1, "kernel size must be odd"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.alpha = nn.Parameter(torch.rand(in_channels, out_channels))
        self.sigma = nn.Parameter(torch.rand(in_channels, out_channels))
        self.exp = nn.Parameter(torch.full((in_channels, out_channels), 2.0))

        self.padding = 0

        # construct the 2D coordinate grid
        ax = torch.linspace(-(kernel_size // 2), (kernel_size // 2), steps=kernel_size)
        xx, yy = torch.meshgrid(ax, ax)
        self.register_buffer("grid_x", xx)
        self.register_buffer("grid_y", yy)

    def forward(self, x):
        # calculate the gaussian filter
        r = torch.sqrt(self.grid_x**2 + self.grid_y**2)

        # ensure proper broadcasting over batches
        alpha = self.alpha.unsqueeze(-1).unsqueeze(-1)
        sigma = self.sigma.unsqueeze(-1).unsqueeze(-1)
        exp = self.exp.unsqueeze(-1).unsqueeze(-1)

        filters = 1 / alpha * torch.exp(-(r**exp) / (sigma**exp))

        # reshape filter to match conv2d weights shape
        filters = filters.view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)

        # perform convolution
        x = F.conv2d(x, filters, padding=self.padding)

        return x


class LithoAct(nn.Module):
    def __init__(self):
        super(LithoAct, self).__init__()
        # Initialize the parameters
        self.alpha = nn.Parameter(torch.tensor(0.25))
        self.beta = nn.Parameter(torch.tensor(0.25))
        self.gamma = nn.Parameter(torch.tensor(0.25))
        self.delta = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        return self.alpha * torch.tanh(self.beta * x - self.delta) + self.gamma * x


class Net(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Net, self).__init__()
        self.conv = GaussianConv2d(in_channels, out_channels, kernel_size)
        self.act = LithoAct()

    def forward(self, x):
        x = self.conv(x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.act(x)
        return x


def train(model, data_loader, criterion, optimizer, epochs, model_count, kernel_size):
    trajectory = []
    trajectory.append((0, model.conv.alpha.item(), model.conv.sigma.item(), model.conv.exp.item()))

    for epoch in range(epochs):
        for inputs, targets in tqdm(data_loader):
            # Transfer inputs to the appropriate device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            pad = (kernel_size - 1) // 2
            targets = F.pad(targets, (-pad, -pad, -pad, -pad))
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Record parameters after each epoch
        trajectory.append([model_count, epoch + 1, model.conv.alpha.item(), model.conv.sigma.item(), model.conv.exp.item()])

    return trajectory


if __name__ == "__main__":
    # define grid
    alphas = np.logspace(-2, 2, num=32)  # alpha values from 0.01 to 100
    sigmas = np.logspace(-2, 2, num=32)  # sigma values from 0.01 to 100

    image_list = os.listdir(path_input)
    train_list = image_list
    # train_list = image_list[: int(len(image_list) * 0.6)]
    # val_list = image_list[int(len(image_list) * 0.6) : int(len(image_list) * 0.8)]
    # test_list = image_list[int(len(image_list) * 0.8) :]

    train_dataset = SEMForwardDataset(
        path_input,
        path_target,
        train_list,
    )
    # val_dataset = SEMForwardDataset(
    #     path_input,
    #     path_target,
    #     val_list,
    # )
    # test_dataset = SEMForwardDataset(
    #     path_input,
    #     path_target,
    #     test_list,
    # )

    train_loader = data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    # val_loader = data.DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)
    # test_loader = data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Define the loss function
    criterion = nn.MSELoss()

    epochs = 10
    model_count = 0
    kernel_size = 201

    for alpha in alphas:
        for sigma in sigmas:
            # Initialize model
            model = Net(in_channels=1, out_channels=1, kernel_size=kernel_size).to(device)

            # Set initial values for alpha and sigma
            model.conv.alpha.data.fill_(alpha)
            model.conv.sigma.data.fill_(sigma)
            model.conv.exp.data.fill_(2.0)  # initial exp value

            # Define optimizer for this model
            optimizer = optim.Adam(model.parameters(), lr=0.1)

            # Training the model and get trajectory
            trajectory = train(model, train_loader, criterion, optimizer, epochs, model_count, kernel_size)

            # Open or create a new csv file
            with open("trajectories.csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(trajectory)

            with open("activations.csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                activations = [model.act.alpha.item(), model.act.beta.item(), model.act.gamma.item(), model.act.delta.item()]
                writer.writerow(activations)
