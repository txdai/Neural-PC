import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, 512), nn.SiLU(), nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, input_shape),
            nn.Sigmoid(),  # use sigmoid as the grayscale value ranges from 0 to 1
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decoder(z)
        return x_reconst, mu, log_var


def vae_loss(x, x_reconst, mu, log_var):
    reconst_loss = nn.functional.binary_cross_entropy(x_reconst, x, reduction="sum")
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return reconst_loss + kl_div * 0.1
