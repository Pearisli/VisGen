import torch
import torch.nn as nn
from typing import Tuple, Type

def normalize(channels: int, momentum: float = 0.1):
    return nn.BatchNorm2d(num_features=channels, momentum=momentum)

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_fn: str = "relu"
    ):
        super().__init__()
        assert act_fn in ("relu", "leaky")

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act_fn = act_fn
        self.nonlinearity = nn.ReLU() if act_fn == "relu" else nn.LeakyReLU(0.2)

        self._init_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.nonlinearity(x)
        return x
    
    def _init_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity=self.act_fn)
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

class DownBlock(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pooling: str = "max"
    ):
        super().__init__(
            nn.MaxPool2d(2) if pooling == "max" else nn.AvgPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

class UpBlock(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__(*[
            nn.Upsample(scale_factor=2),
            ConvBlock(in_channels, out_channels)
        ])

class StackBlock(nn.Sequential):

    def __init__(
        self,
        block_out_channels: Tuple[int, ...],
        block_type: Type[nn.Module],
        **kwargs
    ):
        blocks = []
        for in_channels, out_channels in zip(block_out_channels[:-1], block_out_channels[1:]):
            blocks.append(block_type(in_channels, out_channels, **kwargs))
        super().__init__(*blocks)