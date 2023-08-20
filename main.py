import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from train_backward import run_backward
from test import run_test

input_size = 512
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

for dosage in [1, 2, 3]:
    print(f"Training for dosage {dosage}")
    path_input = f"./data/dosage{dosage}/SEM"
    path_target = f"./data/dosage{dosage}/GDS"

    log_dir = f"logs/dose{dosage}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)

    backward_model, backward_model_name = run_backward(
        path_input,
        path_target,
        device,
        input_size=input_size,
        log_dir=log_dir,
        num_epochs=50,
        writer=writer,
    )

    run_test(dosage, device, latent_dim, patch_size, overlap=overlap, input_size=input_size)

    writer.close()
