import argparse
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from omegaconf import OmegaConf
from tqdm.auto import tqdm

from models import VAE, VAEOutput
from src.dataset.animeface import load_animeface_dataset
from src.util.setting import seed_all, save_checkpoint
from src.util.image_util import run_reconstruction_test, generate_and_save_samples
from src.util.lr_scheduler import get_cosine_schedule_with_warmup

class VAELoss(nn.Module):

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def kld_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    def forward(self, output: VAEOutput, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconst_loss = F.mse_loss(output.sample, target)
        kld_loss = self.kld_loss(output.mean, output.logvar)

        d_input = np.prod(output.sample.shape[1:])
        d_latent = np.prod(output.mean.shape[1:])

        kld_weight = self.beta * (d_latent / d_input)
        loss = reconst_loss + kld_weight * kld_loss

        return {"loss": loss, "reconst": reconst_loss, "kld": kld_loss}

class VAETrainer:

    def __init__(
        self,
        model: VAE,
        loss_fn: VAELoss,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
    
    def train(self, dataloader: DataLoader, num_epochs: int) -> None:
        self.model.train()

        for epoch in range(num_epochs):
            progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", total=len(dataloader))

            for images, _ in progress_bar:

                images = images.to(self.device)
                output: VAEOutput = self.model(images)

                loss_dict = self.loss_fn(output, images)
                loss_dict["loss"].backward()
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                progress_bar.set_postfix({
                    "loss": f"{loss_dict['loss'].item():.4f}",
                    "reconst": f"{loss_dict['reconst'].item():.4f}",
                    "kl_div": f"{loss_dict['kld'].item():.4f}",
                    "lr": f"{self.lr_scheduler.get_last_lr()[0]:.6f}"
                })

def main():
    # ------------- Argument & Config -------------
    parser = argparse.ArgumentParser(description="Train Variational Autoencoder")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_vae.yaml",
        help="Path to config file."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU index"
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    # ------------- Seed setting -------------
    seed_all(cfg.seed)

    # ------------- Model Setup -------------
    hidden_channels = cfg.hidden_channels
    block_out_channels = tuple([hidden_channels * (2 ** i) for i in range(cfg.num_downsample + 1)])
    sample_size = cfg.resolution // (2 ** cfg.num_downsample)

    model = VAE(
        sample_size=sample_size,
        in_channels=3,
        out_channels=3,
        latent_channels=cfg.latent_channels,
        block_out_channels=block_out_channels,
    )

    # ------------- Data Loading -------------
    train_dataloader = load_animeface_dataset(
        base_data_dir=cfg.base_data_dir,
        resolution=cfg.resolution,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        normalize=cfg.normalize
    )

    # ------------- Device, Optimizer & Scheduler -------------
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    num_warmup_steps = len(train_dataloader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=cfg.num_epochs * num_warmup_steps
    )

    # ------------- Training -------------
    trainer = VAETrainer(
        model=model,
        loss_fn=VAELoss(beta=cfg.beta),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device
    )
    trainer.train(train_dataloader, cfg.num_epochs)

    # ------------- Checkpoint Saving -------------
    checkpoint_path = os.path.join(cfg.checkpoint_save_dir, "model.pth")
    save_checkpoint(model, checkpoint_path)

    # ------------- Testing: Reconstruction & Sampling -------------
    os.makedirs(cfg.image_save_dir, exist_ok=True)
    # Reconstruction test
    run_reconstruction_test(
        model=model,
        dataloader=train_dataloader,
        device=device,
        save_path=os.path.join(cfg.image_save_dir, "reconstruction.png"),
        normalize=cfg.normalize,
        nrow=cfg.nrow
    )
    # Sample generation
    generate_and_save_samples(
        model=model,
        device=device,
        save_path=os.path.join(cfg.image_save_dir, "sample.png"),
        seed=cfg.seed,
        normalize=cfg.normalize,
        nrow=cfg.nrow
    )

if __name__ == "__main__":
    main()
