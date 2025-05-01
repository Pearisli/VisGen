import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Union
from tqdm.auto import tqdm

from src.modules.diffusion import (
    DownBlock,
    AttnDownBlock,
    UpBlock,
    AttnUpBlock,
    AttnMidBlock,
    Timesteps,
    TimestepEmbedding,
    normalize
)
import src.util.image_util as image_util

class FlowMatchScheduler:

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0
    ):  
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self.step_index = None
        self.begin_index = None

        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    def _sigma_to_t(self, sigma: float):
        return sigma * self.num_train_timesteps

    def set_timesteps(
        self,
        num_inference_steps: int,
    ) -> None:
        self.num_inference_steps = num_inference_steps
        timesteps = np.linspace(
            self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
        )

        sigmas = timesteps / self.num_train_timesteps

        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
        timesteps = sigmas * self.num_train_timesteps

        self.timesteps = timesteps
        self.sigmas = torch.cat([sigmas, torch.zeros(1)])

        self.step_index = None
        self.begin_index = None

    def index_for_timestep(self, timestep: torch.Tensor, schedule_timesteps: torch.Tensor = None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _initstep_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self.step_index = self.index_for_timestep(timestep)
        else:
            self.step_index = self.begin_index

    def get_sigmas(
        self,
        timesteps: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        sigmas = self.sigmas.to(device)
        schedule_timesteps = self.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < 4:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
    ) -> torch.Tensor:

        if self.step_index is None:
            self._initstep_index(timestep)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output

        self.step_index += 1

        return prev_sample

class UNetModel(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        dropout: float = 0.0,
        down_block_types: Tuple[Union[AttnDownBlock, DownBlock]] = (DownBlock, DownBlock, DownBlock, AttnDownBlock),
        up_block_types: Tuple[Union[AttnUpBlock, UpBlock]] = (AttnUpBlock, UpBlock, UpBlock, UpBlock),
        layers_per_block: int = 1,
        attention_head_dim: int = 8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.dropout = dropout

        # input
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        # time
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0])

        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            self.down_blocks.append(down_block_type(
                input_channel,
                output_channel,
                time_embed_dim,
                dropout=dropout,
                num_layers=layers_per_block,
                add_downsample=not is_final_block,
                attention_head_dim=attention_head_dim,
            ))
        
        # mid
        self.mid_block = AttnMidBlock(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            dropout=dropout,
            num_layers=layers_per_block,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1
            
            self.up_blocks.append(up_block_type(
                input_channel,
                prev_output_channel,
                output_channel,
                time_embed_dim,
                dropout=dropout,
                num_layers=layers_per_block + 1,
                add_upsample=not is_final_block,
                attention_head_dim=attention_head_dim,
            ))
            prev_output_channel = output_channel
        
        # out
        self.conv_norm_out = normalize(self.block_out_channels[0])
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    def get_time_embed(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, int]
    ) -> torch.Tensor:
        timesteps = timestep

        if not isinstance(timestep, torch.Tensor):
            timesteps = torch.tensor([timesteps], dtype=torch.int64)

        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        return t_emb

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:

        # 1. time
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_sample = downsample_block(sample, emb)
            down_block_res_samples += res_sample

        # 4. mid
        sample = self.mid_block(sample, emb)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
    
class FlowMatchPipeline(nn.Module):

    def __init__(
        self,
        sample_size: int,
        unet: UNetModel,
        scheduler: FlowMatchScheduler,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.unet = unet
        self.scheduler = scheduler

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        generator: torch.Generator,
        num_inference_steps: int = 1000,
    ) -> torch.Tensor:
        device = next(self.unet.parameters()).device
        
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps.to(device)

        noisy_latent = torch.randn(
            size=(num_samples, 3, self.sample_size, self.sample_size),
            device=device,
            generator=generator
        )

        bar = tqdm(
            enumerate(timesteps),
            total=len(timesteps),
            leave=True,
            desc="Diffusion denoising"
        )

        for i, t in bar:
            model_pred = self.unet(noisy_latent, t)
            noisy_latent = self.scheduler.step(model_pred, t, noisy_latent)

        sample = noisy_latent
        sample = image_util.denormalize(sample)
        return sample