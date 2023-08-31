import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import large_kernel_block, conv_bn_relu


class Network(nn.Module):
    def __init__(
        self,
        large_kernel_sizes,
        layers,
        channels,
        small_kernel,
        dw_ratio=1,
        ffn_ratio=4,
        input_size=2048,
        device="cuda",
    ):
        super().__init__()
        self.num_stages = len(layers)
        self.input_size = input_size

        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            layer = large_kernel_block(
                channels=channels[stage_idx],
                num_blocks=layers[stage_idx],
                stage_lk_size=large_kernel_sizes[stage_idx],
                small_kernel=small_kernel,
                dw_ratio=dw_ratio,
                ffn_ratio=ffn_ratio,
            )
            self.stages.append(layer)
            if stage_idx < len(layers) - 1:
                transition = nn.Sequential(
                    conv_bn_relu(channels[stage_idx], channels[stage_idx + 1], 1, 1, 0, groups=1),
                    conv_bn_relu(channels[stage_idx + 1], channels[stage_idx + 1], 3, stride=2, padding=1, groups=channels[stage_idx + 1]),
                )
                self.transitions.append(transition)
        self.device = device
        self.to(self.device)

    def forward(self, x):
        for stage_idx in range(self.num_stages):
            x = self.stages[stage_idx](x)
            if stage_idx < self.num_stages - 1:
                x = self.transitions[stage_idx](x)
        return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, "merge_kernel"):
                m.merge_kernel()


class ConvAE(nn.Module):
    def __init__(self, vae_model=None, conv_model=None, latent_size=32, patch_size=24, overlap=8, input_size=528):
        super(ConvAE, self).__init__()
        self.conv_model = conv_model
        self.encoder = vae_model.encoder
        self.decoder = vae_model.decoder
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = self.patch_size - self.overlap
        self.latent_size = latent_size
        self.input_size = input_size

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        x = self.forward_conv(x)
        x = self.forward_decoder(x)
        return x

    def forward_conv(self, x):
        return self.conv_model(x)

    def forward_decoder(self, x: torch.Tensor):
        x = x.view(x.shape[0], self.latent_size, -1).permute(0, 2, 1)
        decoded_patches = self.decoder(x)
        decoded_patches = decoded_patches.permute(0, 2, 1)
        out = torch.nn.functional.fold(
            decoded_patches, (self.input_size + self.stride, self.input_size + self.stride), (self.patch_size, self.patch_size), stride=self.stride
        )
        normalization_map = torch.nn.functional.fold(
            torch.ones_like(decoded_patches),
            (self.input_size + self.stride, self.input_size + self.stride),
            (self.patch_size, self.patch_size),
            stride=self.stride,
        )
        out = out / normalization_map
        out = out[:, :, self.stride // 2 : self.stride // 2 + self.input_size, self.stride // 2 : self.stride // 2 + self.input_size]
        return out
