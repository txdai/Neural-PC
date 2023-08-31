import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
from train_backward import run_backward
from train_ae import run_ae
from models.vae_model import VAE
from test import get_largest_model_number, get_largest_ae_number, run_test

# For debugging
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def after_epoch():
    os.system("./update.sh")


patch_size = 16
latent_dim = 16
overlap = 8
input_size = 512
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

flag_train_ae = False

if flag_train_ae:
    root_dir = f"logs/ae_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.mkdir(root_dir)
    paths = [
        "./data/dosage1/SEM",
        "./data/dosage2/SEM",
        "./data/dosage3/SEM",
        "./data/dosage1/GDS",
        "./data/dosage2/GDS",
        "./data/dosage3/GDS",
        "./data/data",
    ]
    writer = SummaryWriter(root_dir)
    ae_model, ae_model_name = run_ae(
        paths=paths,
        patch_size=patch_size,
        patch_count=32,
        latent_dim=latent_dim,
        num_epochs=50,
        device=device,
        writer=writer,
        log_dir=root_dir,
    )
    writer.close()
else:
    ae_model = VAE(input_shape=patch_size * patch_size, latent_dim=latent_dim).to(device)
    root_dir = get_largest_ae_number("./logs")
    root_dir = "logs/" + root_dir
    ae_model_name = get_largest_model_number(f"./{root_dir}", "ae")
    ae_model.load_state_dict(torch.load(f"./{root_dir}/{ae_model_name}", map_location=device))
    print(f"Loaded model {ae_model_name} from {root_dir}")


for dosage in [1, 2, 3]:
    print(f"Training for dosage {dosage}")
    path_input = f"./data/dosage{dosage}/SEM"
    path_target = f"./data/dosage{dosage}/GDS"

    ae_model = VAE(input_shape=patch_size * patch_size, latent_dim=latent_dim).to(device)
    ae_model.load_state_dict(torch.load(f"./{root_dir}/{ae_model_name}", map_location=device))

    log_dir = f"logs/dose{dosage}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)

    backward_model, backward_model_name = run_backward(
        path_input,
        path_target,
        device,
        input_size=input_size,
        log_dir=log_dir,
        num_epochs_fixed=80,
        num_epochs_ae=50,
        latent_size=latent_dim,
        encoder=ae_model.encoder,
        decoder=ae_model.decoder,
        patch_size=patch_size,
        overlap=overlap,
        writer=writer,
    )

    run_test(dosage, device, latent_dim, patch_size, overlap=overlap, input_size=input_size)

    writer.close()
