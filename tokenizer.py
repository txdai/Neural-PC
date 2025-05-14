import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union


def calculate_patch_sizes(image_size: int = 3000, levels: int = 10) -> Tuple[int, ...]:
    """
    Calculate VAR-style patch sizes for a given image size.
    VAR uses a progression of (1, 2, 3, 4, 5, 6, 8, 10, 13, 16) relative to base patch size.
    """
    # VAR's relative patch numbers
    var_progression = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    # Calculate base patch size to ensure largest patch divides image size
    base_patch = image_size // (var_progression[-1] * 4)  # Divide by 4 to ensure enough detail
    base_patch = max(base_patch, 16)  # Ensure minimum patch size

    # Calculate actual patch sizes
    patch_sizes = tuple(base_patch * p for p in var_progression[:levels])
    return patch_sizes


class BinarySEMVectorQuantizer(nn.Module):
    """Vector quantizer optimized for binary SEM images with VAR-style multi-resolution support."""

    def __init__(
        self,
        image_size: int = 3000,
        vocab_size: int = 4096,
        embedding_dim: int = 32,
        beta: float = 0.25,
        num_levels: int = 10,
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.patch_sizes = calculate_patch_sizes(image_size, num_levels)
        self.quant_resi = quant_resi

        print(f"Using patch sizes: {self.patch_sizes}")

        # Embedding codebook
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1 / vocab_size, 1 / vocab_size)

        # Track codebook usage for each resolution level
        self.register_buffer("ema_vocab_hit", torch.zeros(len(self.patch_sizes), vocab_size))

        # Projection before quantization for each scale
        self.pre_quant = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
                    nn.GroupNorm(8, embedding_dim),
                    nn.GELU(),
                    nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
                )
                for _ in self.patch_sizes
            ]
        )

        # Residual convolution layers with sharing (VAR-style)
        if share_quant_resi > 0:
            self.residual_convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
                        nn.GroupNorm(8, embedding_dim),
                        nn.GELU(),
                        nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
                    )
                    for _ in range(min(share_quant_resi, len(self.patch_sizes)))
                ]
            )
        else:
            self.residual_convs = nn.ModuleList([nn.Identity()])

    def get_residual_conv(self, scale_idx: int) -> nn.Module:
        """Get the appropriate residual conv layer based on scale index."""
        num_convs = len(self.residual_convs)
        if num_convs == 1:
            return self.residual_convs[0]
        # Map scale_idx to available convs using VAR's approach
        conv_idx = min(int(scale_idx * num_convs / (len(self.patch_sizes) - 1)), num_convs - 1)
        return self.residual_convs[conv_idx]

    def quantize(self, z: torch.Tensor, scale_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize encoded patches with scale-specific processing."""
        # Pre-quantization processing
        z = self.pre_quant[scale_idx](z)
        z_flattened = z.reshape(-1, self.embedding_dim)

        # Calculate distances and find nearest embedding
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # Compute losses
        loss = F.mse_loss(z_q.detach(), z) + self.beta * F.mse_loss(z_q, z.detach())

        # Straight through estimator
        z_q = z + (z_q - z).detach()

        return z_q, min_encoding_indices, loss

    def extract_patches(self, x: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Extract patches using VAR's approach."""
        B, C, H, W = x.shape

        # Calculate number of patches
        n_h = H // patch_size
        n_w = W // patch_size

        # Extract patches using unfold
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        patches = patches.view(B, C, patch_size, patch_size, n_h * n_w)
        patches = patches.permute(0, 4, 1, 2, 3).contiguous()  # [B, N, C, H, W]

        return patches, (n_h, n_w)

    def combine_patches(self, patches: torch.Tensor, grid_size: Tuple[int, int], original_size: Tuple[int, int]) -> torch.Tensor:
        """Combine patches back to image using VAR's approach."""
        B, N, C, H, W = patches.shape
        n_h, n_w = grid_size

        # Reshape and fold patches
        patches = patches.permute(0, 2, 3, 4, 1).contiguous()
        patches = patches.view(B, C * H * W, N)

        output = F.fold(patches, output_size=(n_h * H, n_w * W), kernel_size=H, stride=H)

        # Resize to original if needed
        if (n_h * H, n_w * W) != original_size:
            output = F.interpolate(output, size=original_size, mode="bicubic", align_corners=False)

        return output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Forward pass with VAR-style multi-scale tokenization."""
        B, C, H, W = x.shape
        indices_list = []
        total_loss = 0
        accumulation = torch.zeros_like(x)

        # Process each scale
        for scale_idx, patch_size in enumerate(self.patch_sizes):
            # Skip if patch size is too large
            if patch_size > min(H, W):
                continue

            # Extract patches
            patches, grid_size = self.extract_patches(x - accumulation, patch_size)
            B, N, C, pH, pW = patches.shape
            patches = patches.view(B * N, C, pH, pW)

            # Quantize
            z_q, indices, loss = self.quantize(patches, scale_idx)
            indices_list.append(indices.view(B, N))
            total_loss += loss

            # Apply residual processing
            residual_conv = self.get_residual_conv(scale_idx)
            z_q = z_q * (1 - self.quant_resi) + residual_conv(z_q) * self.quant_resi

            # Reshape and combine patches
            z_q = z_q.view(B, N, C, pH, pW)
            decoded = self.combine_patches(z_q, grid_size, (H, W))
            accumulation = accumulation + decoded

            # Update codebook usage statistics
            with torch.no_grad():
                unique_indices, counts = torch.unique(indices, return_counts=True)
                self.ema_vocab_hit[scale_idx, unique_indices] += counts.float()

        return torch.sigmoid(accumulation), indices_list, total_loss


class BinarySEMTokenizer(nn.Module):
    """Complete tokenizer for binary SEM images with VAR-style architecture."""

    def __init__(
        self,
        image_size: int = 3000,
        vocab_size: int = 4096,
        embedding_dim: int = 32,
        num_levels: int = 10,
        quant_resi: float = 0.5,
        share_quant_resi: int = 4,
    ):
        super().__init__()
        self.image_size = image_size

        # Initial processing
        self.process_conv = nn.Sequential(
            nn.Conv2d(1, embedding_dim, 7, padding=3),  # Larger kernel for better feature capture
            nn.GroupNorm(8, embedding_dim),
            nn.GELU(),
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
        )

        # Vector quantizer
        self.quantizer = BinarySEMVectorQuantizer(
            image_size=image_size,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_levels=num_levels,
            quant_resi=quant_resi,
            share_quant_resi=share_quant_resi,
        )

        # Output processing
        self.output_conv = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, 3, padding=1),
            nn.GroupNorm(8, embedding_dim // 2),
            nn.GELU(),
            nn.Conv2d(embedding_dim // 2, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Forward pass through the complete tokenizer."""
        # Ensure input is binary
        x = (x > 0.5).float()

        # Process
        processed = self.process_conv(x)
        quantized, indices, loss = self.quantizer(processed)

        # Generate binary output
        reconstructed = torch.sigmoid(self.output_conv(quantized))

        # Add binary cross entropy loss
        bce_loss = F.binary_cross_entropy(reconstructed, x)
        total_loss = loss + bce_loss

        return reconstructed, indices, total_loss

    @torch.no_grad()
    def encode(self, img: Union[torch.Tensor, np.ndarray]) -> List[torch.Tensor]:
        """Encode a binary image into tokens."""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        img = (img > 0.5).float()

        processed = self.process_conv(img)
        _, indices, _ = self.quantizer(processed)
        return indices

    @torch.no_grad()
    def decode(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """Decode tokens back to binary image."""
        B = indices[0].shape[0]
        accumulation = torch.zeros(B, self.quantizer.embedding_dim, self.image_size, self.image_size, device=indices[0].device)

        for scale_idx, (indices_scale, patch_size) in enumerate(zip(indices, self.quantizer.patch_sizes)):
            B, N = indices_scale.shape

            # Get embeddings and reshape
            z_q = self.quantizer.embedding(indices_scale)  # [B, N, C]
            pH = pW = patch_size
            z_q = z_q.view(B * N, -1, 1, 1).expand(-1, -1, pH, pW)

            # Apply residual refinement
            residual_conv = self.quantizer.get_residual_conv(scale_idx)
            z_q = z_q * (1 - self.quantizer.quant_resi) + residual_conv(z_q) * self.quantizer.quant_resi

            # Reshape and combine
            n = int(np.sqrt(N))
            z_q = z_q.view(B, n, n, -1, pH, pW)
            z_q = z_q.permute(0, 3, 1, 4, 2, 5).contiguous()
            z_q = z_q.view(B, -1, n * pH, n * pW)

            # Resize if needed
            if (n * pH, n * pW) != (self.image_size, self.image_size):
                z_q = F.interpolate(z_q, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)

            accumulation = accumulation + z_q

        # Generate final binary output
        logits = self.output_conv(accumulation)
        reconstructed = torch.sigmoid(logits)
        return (reconstructed > 0.5).float()
