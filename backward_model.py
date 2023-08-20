import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import binarize, compute_signed_distance


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                ConvBlock(in_channels, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
            ]
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.middle = ConvBlock(256, 256)

        self.decoder = nn.ModuleList(
            [
                ConvBlock(256, 256),
                ConvBlock(256, 128),
                ConvBlock(128, 64),
            ]
        )

        self.upconv = nn.ModuleList(
            [
                nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            ]
        )

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.middle(x)

        skip_connections = skip_connections[::-1]

        for i in range(len(self.decoder)):
            x = self.upconv[i](x)
            x = torch.cat([x, skip_connections[i]], dim=1)
            x = self.decoder[i](x)

        x = self.final_conv(x)
        return x


class ConvAE(nn.Module):
    def __init__(self, latent_size=32, encoder=None, decoder=None, patch_size=24, overlap=8, input_size=528):
        super(ConvAE, self).__init__()
        self.unet = UNet(latent_size, latent_size)
        self.encoder = encoder
        self.decoder = decoder
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = self.patch_size - self.overlap
        self.latent_size = latent_size
        self.input_size = input_size

    def forward(self, x: torch.Tensor):
        latent_unfold_shape = self.forward_encoder(x)
        out = self.forward_conv(latent_unfold_shape)
        out = self.forward_decoder(out)
        return out

    def forward_encoder(self, x: torch.Tensor):
        patches = F.unfold(x, (self.patch_size, self.patch_size), stride=self.stride)
        patches = patches.permute(0, 2, 1)
        latent = self.encoder(patches).chunk(2, dim=-1)[0]
        latent = latent.permute(0, 2, 1)
        latent_unfold_shape = latent.reshape(
            x.shape[0], -1, (self.input_size - self.overlap) // self.stride, (self.input_size - self.overlap) // self.stride
        )
        return latent_unfold_shape

    def forward_conv(self, x):
        return self.unet(x)

    def forward_decoder(self, x: torch.Tensor):
        x = x.view(x.shape[0], self.latent_size, -1).permute(0, 2, 1)
        decoded_patches = self.decoder(x)
        decoded_patches = decoded_patches.permute(0, 2, 1)
        out = torch.nn.functional.fold(decoded_patches, (self.input_size, self.input_size), (self.patch_size, self.patch_size), stride=self.stride)

        normalization_map = torch.nn.functional.fold(
            torch.ones_like(decoded_patches), (self.input_size, self.input_size), (self.patch_size, self.patch_size), stride=self.stride
        )
        out = out / normalization_map
        out = torch.clamp(out, min=0.0, max=1.0)
        return out

    def unfreeze_encoder_decoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.decoder.parameters():
            param.requires_grad = True

    def freeze_encoder_decoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False


class ConvSDF(nn.Module):
    def __init__(self, input_size=512):
        super(ConvSDF, self).__init__()
        self.unet = UNet(1, 1)
        self.level = nn.parameter.Parameter(torch.tensor(10.0))
        self.scale = nn.parameter.Parameter(torch.tensor(1.0))
        self.input_size = input_size
        self.relu = nn.LeakyReLU(0.02)

    def forward_conv(self, x):
        return self.unet(x)

    def sdf2img(self, sdf):
        return torch.tanh(self.relu((sdf - self.level) * self.scale))

    def forward(self, x):
        x = compute_signed_distance(binarize(x)).float()
        sdf = self.forward_conv(x)
        out = self.sdf2img(sdf)
        return out
