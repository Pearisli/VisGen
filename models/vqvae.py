import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

from src.modules.resnetblock import (
    StackBlock,
    DownBlock,
    UpBlock,
    normalize
)

class Encoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...] = (64,),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.block_out_channels = block_out_channels

        # Initial convolution layer
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)
        # Down-sampling blocks
        self.down_blocks = StackBlock(block_out_channels, DownBlock)
        # Normalization and activation layers
        self.conv_norm_out = normalize(self.block_out_channels[-1])
        self.conv_act = nn.SiLU()
        # Final convolution to output mean and logvar
        self.conv_out = nn.Conv2d(self.block_out_channels[-1], latent_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(x)
        sample = self.down_blocks(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        
        return sample


class Decoder(nn.Module):

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...] = (64,),
    ) -> None:
        super().__init__()
        self.block_out_channels = list(reversed(block_out_channels))
        self.latent_channels = latent_channels
        self.out_channels = out_channels

        # Reconstruction layers
        self.conv_in = nn.Conv2d(latent_channels, self.block_out_channels[0], kernel_size=3, padding=1)
        self.up_blocks = StackBlock(self.block_out_channels, UpBlock)

        self.conv_norm_out = normalize(self.block_out_channels[-1])
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(self.block_out_channels[-1], out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(z)
        sample = self.up_blocks(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class VectorQuantizer(nn.Module):

    def __init__(
        self,
        n_e: int,
        vq_embed_dim: int,
        beta: float = 0.25
    ):
        super().__init__()
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta

        # Codebook initialization
        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.vq_embed_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        min_encoding_indices = torch.argmin(torch.cdist(z_flattened, self.embedding.weight), dim=1)

        z_q: torch.Tensor = self.embedding(min_encoding_indices).view(z.shape)
        
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        
        # preserve gradients
        z_q: torch.Tensor = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, loss

@dataclass
class VQVAEOutput:

    sample: torch.Tensor
    commit_loss: torch.Tensor

class VQVAE(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        n_e: int = 256,
        block_out_channels: Tuple[int, ...] = (64,),
    ):
        super().__init__()
        self.latent_channels = latent_channels

        self.encoder = Encoder(in_channels, latent_channels, block_out_channels)
        self.decoder = Decoder(out_channels, latent_channels, block_out_channels)

        self.quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.quantize = VectorQuantizer(n_e, latent_channels, beta=0.25)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quant, commit_loss = self.quantize(h)
        quant = self.post_quant_conv(quant)
        
        dec = self.decoder(quant)
        return dec, commit_loss
    
    def forward(self, sample: torch.Tensor) -> VQVAEOutput:
        h = self.encode(sample)
        dec, commit_loss = self.decode(h)
        return VQVAEOutput(dec, commit_loss)

    @torch.no_grad()
    def reconstruct(self, samples: torch.Tensor) -> torch.Tensor:
        return self(samples).sample