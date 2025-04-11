import torch
import torch.nn as nn
from typing import Tuple, Type

def normalize(channels: int, num_groups: int = 16):
    return nn.GroupNorm(num_channels=channels, num_groups=num_groups, eps=1e-6, affine=True)

class ResnetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.norm2 = normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states)

        return output_tensor
    
class Upsample(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(in_channels, out_channels,
                kernel_size=3, stride=1, padding=1, bias=True
            )
        )

class Downsample(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__(
            nn.Conv2d(in_channels, out_channels,
                kernel_size=3, stride=2, padding=1, bias=True
            )
        )

class DownBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        add_downsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsampler = Downsample(out_channels, out_channels)
        else:
            self.downsampler = None
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)
        
        return hidden_states

class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        add_upsample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsampler = Upsample(out_channels, out_channels)
        else:
            self.upsampler = None
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)
        
        return hidden_states

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