import argparse
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from tqdm.auto import tqdm

from models import VQVAE, VQVAEOutput
from src.dataset.animeface import load_animeface_dataset
from src.util.setting import seed_all, save_checkpoint
from src.util.image_util import run_reconstruction_test
from src.util.lr_scheduler import get_cosine_schedule_with_warmup

class VQVAELoss(nn.Module):

    def __init__(self, lam: float = 0.1):
        super().__init__()
        self.lam = lam  # Weight for VQ loss

    def forward(self, output: VQVAEOutput, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconst_loss = F.mse_loss(output.sample, target)
        total_loss = reconst_loss + self.lam * output.commit_loss
        return {
            "loss": total_loss,
            "reconst": reconst_loss,
            "commit_loss": output.commit_loss
        }

class VQVAETrainer:

    def __init__(
        self,
        model: VQVAE,
        loss: VQVAELoss,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
    
    def train(self, dataloader: DataLoader, num_epochs: int):
        self.model.train()

        for epoch in range(num_epochs):
            progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", total=len(dataloader))
            
            for images, _ in progress_bar:

                images = images.to(self.device)
                output = self.model(images)
                
                loss_dict = self.loss(output, images)
                loss_dict["loss"].backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                progress_bar.set_postfix({
                    "loss": f"{loss_dict['loss'].item():.4f}",
                    "reconst": f"{loss_dict['reconst'].item():.4f}",
                    "commit_loss": f"{loss_dict['commit_loss'].item():.4f}",
                    "lr": f"{self.lr_scheduler.get_last_lr()[0]:.6f}"
                })

def main():
    # ------------- Argument & Config -------------
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_vqvae.yaml",
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

    model = VQVAE(
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
    trainer = VQVAETrainer(
        model=model,
        loss=VQVAELoss(),
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

if __name__ == "__main__":
    main()