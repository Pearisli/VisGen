import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import math

def normalize(channels: int):
    return nn.GroupNorm(num_channels=channels, num_groups=32, eps=1e-6, affine=True)

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

class Timesteps(nn.Module):

    def __init__(
        self,
        num_channels: int,
        scale: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.scale = scale

    def forward(
        self,
        timesteps: torch.Tensor,
        max_period: int = 10000,
        downscale_freq_shift: int = 1,
    ) -> torch.Tensor:
        assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

        half_dim = self.num_channels // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - downscale_freq_shift)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        # concat sine and cosine embeddings
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)

        # zero pad
        if self.num_channels % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

class TimestepEmbedding(nn.Module):

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, sample_proj_bias)

    def forward(self, sample: torch.Tensor):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample

class SelfAttention(nn.Module):

    def __init__(
        self,
        attention_dim: int,
        heads: int
    ):
        super().__init__()
        self.heads = heads
        self.attention_dim = attention_dim
        # For q, k, v in one operation
        self.qkv = nn.Conv1d(attention_dim, attention_dim * 3, 1)
        # Final projection layer
        self.proj_out = nn.Conv1d(attention_dim, attention_dim, 1)
        # Scale factor for dot-product attention
        # The dimension per head is attention_dim // heads
        self.scale = (attention_dim // heads) ** -0.5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width)
        # Compute queries, keys, and values with one convolution
        qkv = self.qkv(hidden_states)  # shape: (B, 3 * attention_dim, L)
        # Split the combined tensor into three parts along the channel dimension
        q, k, v = torch.chunk(qkv, 3, dim=1)  # each has shape: (B, attention_dim, L)

        # Reshape into (B, heads, dim_head, L)
        dim_head = channel // self.heads
        q = q.view(batch_size, self.heads, dim_head, height * width)
        k = k.view(batch_size, self.heads, dim_head, height * width)
        v = v.view(batch_size, self.heads, dim_head, height * width)

        # Rearrange for dot-product: (B, heads, L, dim_head)
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        # Compute scaled dot-product attention scores: (B, heads, L, L)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply the attention to the values: (B, heads, L, dim_head)
        out = torch.matmul(attn, v)

        # Rearrange back: (B, heads, dim_head, L) and then combine the head dimension
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, self.attention_dim, height * width)

        # Apply final projection
        out = self.proj_out(out)
        return out

class AttentionBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        heads: int,
        dim_head: int
    ):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.norm = normalize(in_channels)
        self.attn = SelfAttention(in_channels, heads)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states: torch.Tensor = self.norm(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = hidden_states.reshape(residual.shape)

        hidden_states = hidden_states + residual
        return hidden_states

class ResnetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temb_channels = temb_channels

        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        
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

    def forward(self, input_tensor: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        temb = self.nonlinearity(temb)
        temb = self.time_emb_proj(temb)[:, :, None, None]
        hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states)

        return output_tensor

class DownBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        add_downsample: bool = True,
        attention_head_dim: int = 1
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    dropout=dropout,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsampler = Downsample(out_channels, out_channels)
        else:
            self.downsampler = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states = output_states + (hidden_states,)
        
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)
            output_states = output_states + (hidden_states,)
        
        return hidden_states, output_states

class AttnDownBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        add_downsample: bool = True,
        attention_head_dim: int = 1,
    ):
        super().__init__()
        resnets = []
        attentions = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    dropout=dropout,
                )
            )
            attentions.append(
                AttentionBlock(
                    in_channels=out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        if add_downsample:
            self.downsampler = Downsample(out_channels, out_channels)
        else:
            self.downsampler = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        
        output_states = ()

        for (resnet, attn) in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states = output_states + (hidden_states,)
        
        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)
            output_states = output_states + (hidden_states,)
        
        return hidden_states, output_states

class UpBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        add_upsample: bool = True,
        attention_head_dim: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    dropout=dropout,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsampler = Upsample(out_channels, out_channels)
        else:
            self.upsampler = None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: torch.Tensor
    ) -> torch.Tensor:
        
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
        
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)
        
        return hidden_states

class AttnUpBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        add_upsample: bool = True,
        attention_head_dim: int = 1,
    ):
        super().__init__()
        resnets = []
        attentions = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                )
            )
            attentions.append(
                AttentionBlock(
                    in_channels=out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)

        if add_upsample:
            self.upsampler = Upsample(out_channels, out_channels)
        else:
            self.upsampler = None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: torch.Tensor
    ) -> torch.Tensor:
        
        for (resnet, attn) in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
        
        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)
        
        return hidden_states

class AttnMidBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        attention_head_dim: int = 1,
    ):
        super().__init__()
        resnets = [
            ResnetBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                dropout=dropout,
            )
        ]
        attentions = []

        for i in range(num_layers):
            attentions.append(
                AttentionBlock(
                    in_channels=in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_head=attention_head_dim
                )
            )

            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    dropout=dropout,
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
    
    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:

        hidden_states = self.resnets[0](hidden_states, temb)

        for (attn, resnet) in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, temb)
        
        return hidden_states
