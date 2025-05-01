import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from typing import Optional, Tuple, Union

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

class DDPMScheduler:

    def __init__(
        self,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_train_timesteps: int = 1000,
        prediction_type: str = "epsilon"
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

    def __len__(self):
        return self.num_train_timesteps
    
    def _get_variance(self, t: int) -> torch.Tensor:
        prev_t = t - 1

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        return variance

    def _get_effective_tensors(
        self,
        timesteps: torch.IntTensor,
        ref_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.alphas_cumprod = self.alphas_cumprod.to(device=ref_tensor.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=ref_tensor.dtype)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(ref_tensor.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(ref_tensor.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        return sqrt_alpha_prod, sqrt_one_minus_alpha_prod
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        timesteps = timesteps.to(original_samples.device)
        sqrt_alpha_prod, sqrt_one_minus_alpha_prod = self._get_effective_tensors(timesteps, original_samples)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor
    ) -> torch.Tensor:
        timesteps = timesteps.to(sample.device)
        sqrt_alpha_prod, sqrt_one_minus_alpha_prod = self._get_effective_tensors(timesteps, sample)
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
    ) -> torch.Tensor:
        t = timestep

        prev_t = t - 1

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )
        
        # 3. Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 4. Compute predicted previous sample µ_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 5. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = torch.randn(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            variance = (self._get_variance(t) ** 0.5) * variance_noise
        
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

class UNetModel(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        dropout: float = 0.0,
        down_block_types: Tuple[Union[AttnDownBlock, DownBlock]] = (DownBlock, DownBlock, DownBlock, AttnDownBlock),
        up_block_types: Tuple[Union[AttnUpBlock, UpBlock]] = (AttnUpBlock, UpBlock, UpBlock, UpBlock),
        layers_per_block: int = 2,
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
            self.block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    def get_time_embed(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, int]
    ) -> torch.Tensor:
        timesteps = timestep

        if not isinstance(timestep, torch.Tensor):
            timesteps = torch.tensor([timesteps], dtype=torch.int64)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
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
    
class DDPMPipeline(nn.Module):

    def __init__(
        self,
        sample_size: int,
        unet: UNetModel,
        scheduler: DDPMScheduler,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.unet = unet
        self.scheduler = scheduler

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 1000
    ) -> torch.Tensor:
        assert num_inference_steps == self.scheduler.num_train_timesteps
        device = next(self.unet.parameters()).device

        timesteps = self.scheduler.timesteps.to(device)

        noisy_latent = torch.randn(
            size=(num_samples, 3, self.sample_size, self.sample_size),
            device=device,
            generator=generator
        )

        progress_bar = tqdm(
            enumerate(timesteps),
            total=len(timesteps),
            leave=True,
            desc="Diffusion denoising"
        )

        for _, t in progress_bar:
            model_pred = self.unet(noisy_latent, t)
            noisy_latent = self.scheduler.step(model_pred, t, noisy_latent)

        sample = noisy_latent
        sample = image_util.denormalize(sample)
        return sample