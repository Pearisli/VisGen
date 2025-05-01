import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

from src.modules.resnetblock import (
    StackBlock,
    DownBlock,
    UpBlock,
    normalize
)
import src.util.image_util as image_util

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
        self.conv_in = nn.Conv2d(in_channels, self.block_out_channels[0], kernel_size=3, padding=1)
        # Down-sampling blocks
        self.down_blocks = StackBlock(self.block_out_channels, DownBlock)
        # Normalization and activation layers
        self.conv_norm_out = normalize(self.block_out_channels[-1])
        self.conv_act = nn.SiLU()
        # Final convolution to output mean and logvar
        self.conv_out = nn.Conv2d(self.block_out_channels[-1], 2 * latent_channels, kernel_size=3, padding=1)

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
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.block_out_channels = list(reversed(block_out_channels))

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


@dataclass
class VAEOutput:

    sample: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor


class VAE(nn.Module):

    def __init__(
        self,
        sample_size: int,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        block_out_channels: Tuple[int, ...] = (64,),
    ):
        super().__init__()
        self.sample_size = sample_size
        self.latent_channels = latent_channels

        self.encoder = Encoder(in_channels, latent_channels, block_out_channels)
        self.decoder = Decoder(out_channels, latent_channels, block_out_channels)

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec 

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, sample: torch.Tensor) -> VAEOutput:
        mean, logvar = self.encode(sample)
        z = self.reparameterize(mean, logvar)
        dec = self.decode(z)
        return VAEOutput(dec, mean, logvar)

    @torch.no_grad()
    def reconst(self, images: Union[torch.Tensor, List[torch.Tensor]], normalize: bool = False) -> torch.Tensor:
        if isinstance(images, List):
            images = torch.stack(images)

        device = next(self.parameters()).device

        inputs = images.to(device)
        if normalize:
            inputs = image_util.normalize(inputs)

        outputs = self(inputs).sample
        outputs = image_util.denormalize(outputs)
        return outputs

    @torch.no_grad()
    def sample(self, num_samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        z = torch.randn(
            size=(num_samples, self.latent_channels, self.sample_size, self.sample_size),
            device=device,
            generator=generator
        )
        x = self.decode(z)
        x = image_util.denormalize(x)
        return x