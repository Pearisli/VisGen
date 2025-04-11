import torch
import torch.nn as nn
from typing import Tuple, Optional

from src.modules.resnetblock import (
    StackBlock,
    DownBlock,
    UpBlock,
    normalize
)

class DownBlock(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__(
            nn.Conv2d(in_channels, out_channels,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

class Generator(nn.Module):

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 128,
        block_out_channels: Tuple[int, ...] = (64, )
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.block_out_channels = list(reversed(block_out_channels))

        self.conv_in = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, self.block_out_channels[0], kernel_size=4, stride=1, padding=0),
            normalize(self.block_out_channels[0]),
            nn.SiLU()
        )
        self.up_blocks = StackBlock(self.block_out_channels, UpBlock)
        self.conv_norm_out = normalize(self.block_out_channels[-1])
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(self.block_out_channels[-1], out_channels, kernel_size=3, padding=1)

        self.nonlinearity = nn.Tanh()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(z)
        hidden_states = self.up_blocks(hidden_states)
        x = self.conv_out(hidden_states)
        x = self.nonlinearity(x)
        return x

class Critic(nn.Module):

    def __init__(
        self,
        in_channels: int,
        block_out_channels: Tuple[int, ...] = (64,),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.block_out_channels = block_out_channels

        self.conv_in = nn.Conv2d(in_channels, self.block_out_channels[0], kernel_size=3, padding=1)
        self.down_blocks = StackBlock(self.block_out_channels, DownBlock)
        self.conv_out = nn.Conv2d(self.block_out_channels[-1], 1, kernel_size=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(x)
        hidden_states = self.down_blocks(hidden_states)
        logits = self.conv_out(hidden_states)
        logits = logits.view(-1, 1)
        return logits

class WGAN(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 256,
        block_out_channels: Tuple[int, ...] = (64,),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.block_out_channels = block_out_channels

        self.generator = Generator(
            out_channels=out_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels
        )

        self.critic = Critic(
            in_channels=in_channels,
            block_out_channels=block_out_channels
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def get_noise(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        return torch.randn((batch_size, self.latent_channels, 1, 1), device=device, generator=generator)
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        noise = self.get_noise(num_samples, device, generator)
        out = self.generator(noise)
        return out