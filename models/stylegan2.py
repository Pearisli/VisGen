import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from typing import List, Tuple, Optional

from src.modules.resnetblock import (
    StackBlock,
)

# Modified from https://blog.paperspace.com/implementation-stylegan2-from-scratch/

class EqualizedWeight(nn.Module):

    def __init__(
        self,
        shape,
    ):
        super().__init__()
        self.c = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))
    
    def forward(self) -> torch.Tensor:
        return self.weight * self.c

class EqualizedLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: float = 0.
    ):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight(), bias=self.bias)

class EqualizedConv2d(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        padding: int = 0
    ):
        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)

class Conv2dWeightModulate(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        demodulate: bool = True,
        eps = 1e-8
    ):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2

        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor
    ) -> torch.Tensor:

        b, _, h, w = x.shape

        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)

class MappingNetwork(nn.Module):

    def __init__(
        self,
        z_dim: int,
        w_dim: int,
        num_layers: int = 8
    ):
        super().__init__()
        self.mapping = nn.Sequential(
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU()
        )
        for i in range(num_layers - 1):
            self.mapping.append(EqualizedLinear(w_dim, w_dim))
            self.mapping.append(nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)  # for PixelNorm
        return self.mapping(x)

class StyleBlock(nn.Module):

    def __init__(
        self,
        w_dim: int,
        in_features: int,
        out_features: int
    ):
        super().__init__()

        self.to_style = EqualizedLinear(w_dim, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.nonlinearity = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise: torch.Tensor = None
    ) -> torch.Tensor:
        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.nonlinearity(x + self.bias[None, :, None, None])

class ToRGB(nn.Module):

    def __init__(
        self, 
        w_dim: int,
        features: int
    ):
        super().__init__()
        self.to_style = EqualizedLinear(w_dim, features, bias=1.0)

        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.nonlinearity = nn.LeakyReLU(0.2)

    def forward(
        self, 
        x: torch.Tensor,
        w: torch.Tensor
    ) -> torch.Tensor:
        style = self.to_style(w)
        x = self.conv(x, style)
        return self.nonlinearity(x + self.bias[None, :, None, None])

# Generator
class StyleBlock(nn.Module):

    def __init__(
        self,
        w_dim: int,
        in_features: int,
        out_features: int
    ):
        super().__init__()

        self.to_style = EqualizedLinear(w_dim, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.nonlinearity = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise: torch.Tensor = None
    ) -> torch.Tensor:
        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.nonlinearity(x + self.bias[None, :, None, None])

class GeneratorBlock(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        w_dim: int,
    ):
        super().__init__()

        self.style_block1 = StyleBlock(w_dim, in_features, out_features)
        self.style_block2 = StyleBlock(w_dim, out_features, out_features)

        self.to_rgb = ToRGB(w_dim, out_features)

    def forward(
        self, 
        x: torch.Tensor,
        w: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])

        rgb = self.to_rgb(x, w)

        return x, rgb

class Generator(nn.Module):

    def __init__(
        self,
        log_resolution: int,
        w_dim: int,
        n_features: int = 32,
        max_features: int = 256,
    ):
        super().__init__()

        block_out_channels = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        reversed_block_out_channels = list(reversed(block_out_channels))
        self.n_blocks = len(reversed_block_out_channels)

        self.initial_constant = nn.Parameter(torch.randn((1, reversed_block_out_channels[0], 4, 4)))

        self.style_block = StyleBlock(w_dim, reversed_block_out_channels[0], reversed_block_out_channels[0])
        self.to_rgb = ToRGB(w_dim, reversed_block_out_channels[0])

        self.blocks = StackBlock(reversed_block_out_channels, GeneratorBlock, w_dim=w_dim)

    def forward(
        self,
        w: torch.Tensor, 
        input_noise: torch.Tensor
    ) -> torch.Tensor:
        batch_size = w.shape[1]

        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])

        for i in range(1, self.n_blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear") + rgb_new

        return torch.tanh(rgb)

# Discriminator
class DiscriminatorBlock(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int
    ):
        super().__init__()
        self.residual = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2), # down sampling using avg pool
            EqualizedConv2d(in_features, out_features, kernel_size=1)
        )

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.down_sample = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  # down sampling using avg pool

        self.scale = 1 / sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        x = self.block(x)
        x = self.down_sample(x)

        return (x + residual) * self.scale

class Discriminator(nn.Module):

    def __init__(
        self,
        log_resolution: int,
        n_features: int = 32,
        max_features: int = 256
    ):
        super().__init__()

        block_out_channels = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]

        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, block_out_channels[0], 1),
            nn.LeakyReLU(0.2),
        )
        self.blocks = StackBlock(block_out_channels, DiscriminatorBlock)

        final_features = block_out_channels[-1] + 1
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        self.final = EqualizedLinear(final_features * 2 * 2, 1)

    def minibatch_std(self, x: torch.Tensor) -> torch.Tensor:
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.from_rgb(x)
        x = self.blocks(x)

        x = self.minibatch_std(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)

class StyleGAN2(nn.Module):

    def __init__(
        self,
        log_resolution: int,
        latent_channels: int,
        w_dim: int,
        mapping_layers: int = 8,
        n_features: int = 32
    ):
        super().__init__()

        self.log_resolution = log_resolution
        self.latent_channels = latent_channels
        self.w_dim = w_dim

        self.mapping_network = MappingNetwork(latent_channels, w_dim, mapping_layers)
        self.generator = Generator(
            log_resolution=log_resolution,
            w_dim=w_dim,
            n_features=n_features
        )
        self.critic = Discriminator(
            log_resolution=log_resolution,
            n_features=n_features * 2
        )
    
    def get_w(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        z = torch.randn(batch_size, self.w_dim, device=device, generator=generator)
        w = self.mapping_network(z)
        return w[None, :, :].expand(self.log_resolution, -1, -1)

    def get_noise(
        self,
        batch_size: int,
        device: torch.device
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        noise = []
        resolution = 4

        for i in range(self.log_resolution):
            n1 = torch.randn((batch_size, 1, resolution, resolution), device=device) if i > 0 else None
            n2 = torch.randn((batch_size, 1, resolution, resolution), device=device)
            noise.append((n1, n2))

            resolution *= 2

        return noise

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        w = self.get_w(num_samples, device, generator)
        noise = self.get_noise(num_samples, device)
        out = self.generator(w, noise)
        return out