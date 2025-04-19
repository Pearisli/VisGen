import argparse
import math
import os
from copy import deepcopy
from contextlib import contextmanager
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from omegaconf import OmegaConf

from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from diffusers.image_processor import VaeImageProcessor

from models import UNet2DModel
from src.util.setting import seed_all

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train latent diffusion model with VAE and UNet"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    return parser.parse_args()

def setup_directories(base_output: str, job_name: str) -> Dict[str, str]:
    out_run = os.path.join(base_output, job_name)
    dirs = {
        "run": out_run,
        "ckpt": os.path.join(out_run, "checkpoint"),
        "tb": os.path.join(out_run, "tensorboard"),
        "vis": os.path.join(out_run, "visualization"),
    }
    return dirs

class PosteriorDataset(Dataset):

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.latent_paths = [
            os.path.join(self.root_dir, filename)
            for filename in os.listdir(root_dir) if filename.endswith('.npy')
        ]

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, idx: int):
        latent_np = np.load(self.latent_paths[idx])
        latent_pt = torch.from_numpy(latent_np)
        return latent_pt

class LDMPipeline(DiffusionPipeline):

    def __init__(self, vae: AutoencoderKL, unet: UNet2DModel, scheduler: DDIMScheduler):
        super().__init__()
        self.register_modules(vae=vae, unet=unet, scheduler=scheduler)
        # Scale latents to match VAE standard deviation
        self.scale = 0.19805

    @torch.no_grad()
    def sample_from_posterior(
        self, posterior: torch.Tensor, mode: bool = False
    ) -> torch.Tensor:
        mean, std = posterior.chunk(2, dim=1)
        if mode:
            latent = mean
        else:
            # add noise for stochastic sampling
            noise = torch.randn_like(mean)
            latent = mean + std * noise
        # rescale before decoding
        return latent * self.scale

    @torch.no_grad()
    def __call__(
        self,
        num_samples: int,
        num_inference_steps: int,
        sample_size: int,
        generator: Optional[torch.Generator] = None,
        show_bar: bool = False
    ) -> torch.Tensor:
        # initialize noise latents
        latent = torch.randn(
            size=(num_samples, 4, sample_size, sample_size),
            device=self.device, dtype=self.dtype, generator=generator
        )
        # prepare noise_scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps.to(self.device)

        iterator = tqdm(timesteps, disable=not show_bar, desc="    Diffusion denoising", leave=False)
        for t in iterator:
            model_output = self.unet(latent, t).sample
            latent = self.scheduler.step(model_output, t, latent, generator=generator).prev_sample

        # decode and denormalize images
        decoded = self.vae.decode(latent / self.scale).sample
        return VaeImageProcessor.denormalize(decoded)

class LatentDiffusionTrainer:

    def __init__(
        self,
        cfg: OmegaConf,
        accelerator: Accelerator,
        model: LDMPipeline,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler,
        dirs: Dict[str, str],
    ):
        self.cfg = cfg
        self.accelerator = accelerator
        self.device = accelerator.device
        self.dirs = dirs

        # model components
        self.model = model
        self.unet: UNet2DModel = model.unet
        self.vae: AutoencoderKL = model.vae
        self.noise_scheduler: DDIMScheduler = model.scheduler

        # dataloaders and optimizers
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # training settings
        self.max_train_steps = cfg.max_train_steps
        self.num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.prediction_type = self.noise_scheduler.config.prediction_type

        # metrics
        self.best_loss = float("inf")

        # disable gradients for VAE
        self.vae.requires_grad_(False)
        self.unet.enable_xformers_memory_efficient_attention()
        # EMA wrapper for UNet
        self.ema_unet = EMAModel(
            deepcopy(self.unet).parameters(),
            model_cls=UNet2DModel,
            model_config=self.unet.config,
            foreach=True
        )
        self.ema_unet.to(self.device)

        # select dtype
        self.dtype = torch.float32
        if self.cfg.mixed_precision == "fp16":
            self.dtype = torch.float16
        elif self.cfg.mixed_precision == "bf16":
            self.dtype = torch.bfloat16

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("tensorboard")

    @contextmanager
    def ema_scope(self):
        self.ema_unet.store(self.unet.parameters())
        self.ema_unet.copy_to(self.unet.parameters())
        yield None
        self.ema_unet.restore(self.unet.parameters())

    def prepare(self) -> None:
        (
            self.unet,
            self.train_dataloader,
            self.valid_dataloader,
            self.optimizer,
            self.lr_scheduler
        ) = self.accelerator.prepare(
            self.unet,
            self.train_dataloader,
            self.valid_dataloader,
            self.optimizer,
            self.lr_scheduler
        )
        self.vae.to(self.device, self.dtype)

    def compute_loss(self, latents: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        latent = self.model.sample_from_posterior(latents)
        noise = torch.randn_like(latent)
        batch_size = latent.size(0)

        # sample random timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        )
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
        model_output: torch.Tensor = self.unet(noisy_latent, timesteps, return_dict=False)[0]

        # choose target based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = latent
        elif self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latent, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")

        return F.mse_loss(model_output.float(), target.float(), reduction=reduction)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        generator: Optional[torch.Generator] = None,
        show_bar: bool = False
    ) -> torch.Tensor:
        with torch.autocast(self.device.type):
            samples = self.model(
                num_samples=num_samples,
                num_inference_steps=self.cfg.num_inference_steps,
                sample_size=self.cfg.sample_size,
                generator=generator,
                show_bar=show_bar,
            )
        return samples

    @torch.no_grad()
    def validate(self, step: int) -> float:
        self.unet.eval()
        self.vae.eval()

        with torch.autocast(self.device.type):
            # compute validation loss
            total_loss, count = 0.0, 0
            for batch in tqdm(self.valid_dataloader, desc=f"Validation Steps {step}", leave=True):
                loss = self.compute_loss(batch, reduction="none")
                loss = loss.mean(dim=(1, 2, 3))
                total_loss += loss.sum().item()
                count += batch.size(0)
        
        torch.cuda.empty_cache()
        return total_loss / count

    def save_samples(self, samples: torch.Tensor, step: int):
        grid = make_grid(samples, nrow=self.cfg.nrow)
        arr: np.ndarray = grid.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray((arr * 255).astype(np.uint8))
        img.save(os.path.join(self.dirs['vis'], f"iter{step:06d}.jpg"))

    def save_checkpoint(self, ckpt_name: str):
        self.model.unet = self.accelerator.unwrap_model(self.unet)
        self.model.save_pretrained(os.path.join(self.dirs['ckpt'], ckpt_name))

    def train(self):
        self.prepare()

        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.accelerator.gradient_accumulation_steps
        )
        max_epoch = math.ceil(self.max_train_steps / num_update_steps_per_epoch)

        global_step = 0
        progress_bar = tqdm(
            range(self.max_train_steps),
            desc="Training Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(max_epoch):
            train_loss = 0.0
            for batch in self.train_dataloader:
                self.unet.train()

                with self.accelerator.accumulate(self.unet):
                    batch_size = batch.shape[0]

                    loss = self.compute_loss(batch)
                    avg_loss: torch.Tensor = self.accelerator.gather(loss.repeat(batch_size)).mean()

                    train_loss += avg_loss.item() / self.accelerator.gradient_accumulation_steps

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    
                    self.ema_unet.step(self.unet.parameters())
                    progress_bar.update(1)
                    global_step += 1

                    self.accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if self.accelerator.is_main_process:

                        # log and checkpoint at intervals
                        if global_step % self.cfg.validation_steps == 0:
                            with self.ema_scope():
                                val_loss = self.validate(step=global_step)
                                self.accelerator.log({"valid_loss": val_loss}, step=global_step)

                                if val_loss < self.best_loss:
                                    self.best_loss = val_loss
                                    self.save_checkpoint(f"iter{global_step:06d}")
                                
                                # sample fixed grid
                                generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
                                samples = self.sample(num_samples=self.cfg.nrow**2, generator=generator, show_bar=True)
                                self.save_samples(samples, global_step)
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.lr_scheduler.get_last_lr()[0]:.6f}"
                })

                if global_step >= self.max_train_steps:
                    break

def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    seed_all(cfg.seed)

    # Create output directories
    job_name = os.path.basename(args.config).split(".")[0]
    dirs = setup_directories(cfg.output_dir, job_name)

    # initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=dirs['run'], logging_dir=dirs['tb'])
    )

    if accelerator.is_main_process:
        for path in dirs.values():
            os.makedirs(path, exist_ok=False)

    # -------------------- Device --------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # -------------------- Data --------------------
    # prepare dataset and dataloaders
    dataset = PosteriorDataset(cfg.data_dir)
    assert cfg.train_size + cfg.valid_size == len(dataset), "Dataset split mismatch"
    train_dataset, valid_dataset = random_split(dataset, [cfg.train_size, cfg.valid_size])
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                              num_workers=16, pin_memory=True, persistent_workers=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.valid_batch_size, shuffle=False,
                              num_workers=16, pin_memory=True, persistent_workers=True)

    # -------------------- Model --------------------
    if cfg.resume:
        pipeline: LDMPipeline = LDMPipeline.from_pretrained(cfg.pretrained_path, local_files_only=True)
        pipeline.scheduler.register_to_config(prediction_type=cfg.prediction_type)
    else:
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-2", subfolder="vae", local_files_only=True
        )
        unet = UNet2DModel(
            sample_size=cfg.sample_size,
            in_channels=4,
            out_channels=4,
            down_block_types=tuple(cfg.down_block_types),
            up_block_types=tuple(cfg.up_block_types),
            block_out_channels=tuple(cfg.block_out_channels),
            num_attention_heads=tuple(cfg.num_attention_heads),
            layers_per_block=cfg.layers_per_block,
            resnet_time_scale_shift=cfg.resnet_time_scale_shift,
        )
        noise_scheduler = DDIMScheduler(
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            beta_schedule=cfg.beta_schedule,
            prediction_type=cfg.prediction_type,
            timestep_spacing="trailing",
            clip_sample=False
        )
        pipeline = LDMPipeline(vae=vae, unet=unet, noise_scheduler=noise_scheduler)

    # -------------------- Optimizer and Learning Rate Scheduler --------------------
    if cfg.optimizer == "adam":
        optimizer = optim.Adam(pipeline.unet.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer == "adamw":
        optimizer = optim.AdamW(pipeline.unet.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {cfg.optimizer}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.num_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes
    )

    # -------------------- Start Training --------------------
    trainer = LatentDiffusionTrainer(
        cfg=cfg,
        accelerator=accelerator,
        model=pipeline,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dirs=dirs
    )
    trainer.train()

    # Save the checkpoint only on the main process to avoid duplicates.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        trainer.ema_unet.copy_to(trainer.unet.parameters())
        trainer.save_checkpoint("last_iter")
    accelerator.end_training()

if __name__ == "__main__":
    main()
