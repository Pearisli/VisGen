import argparse
import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from omegaconf import OmegaConf
from tqdm.auto import tqdm
from datasets import load_dataset

from models import VAE, VAEOutput
from src.util.setting import seed_all
from src.util.lr_scheduler import get_cosine_schedule_with_warmup
from src.util.image_util import make_grid

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML config file"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Compute device to use, e.g., 'cuda:0', 'cuda:1', or 'cpu'."
    )
    return parser.parse_args()

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

def main():
    # ------------- Config -------------
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    # ------------- Seed -------------
    seed_all(cfg.seed)

    # ------------- Model -------------
    model = VAE(
        sample_size=cfg.sample_size,
        in_channels=3,
        out_channels=3,
        latent_channels=cfg.latent_channels,
        block_out_channels=tuple(cfg.block_out_channels),
    )

    loss_fn = VAELoss(beta=cfg.beta)

    # ------------- Data -------------
    train_dataset = load_dataset(cfg.base_data_dir)["train"]

    train_transforms = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [train_transforms(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    train_dataset.set_transform(preprocess_train)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        drop_last=True,
    )

    # ------------- Device, Optimizer & Scheduler -------------
    device = torch.device(args.device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    num_epochs = cfg.num_epochs
    num_warmup_steps = len(train_dataloader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_epochs * num_warmup_steps
    )

    # ------------- Training -------------
    model.train()
    model.to(device)

    for epoch in range(num_epochs):
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch [{epoch+1}/{num_epochs}]",
            total=len(train_dataloader)
        )

        for batch in progress_bar:

            images = batch["images"].to(device)
            output: VAEOutput = model(images)

            loss_dict = loss_fn(output, images)
            loss_dict["loss"].backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({
                "loss": f"{loss_dict['loss'].item():.4f}",
                "reconst": f"{loss_dict['reconst'].item():.4f}",
                "kl_div": f"{loss_dict['kld'].item():.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}"
            })

    # ------------- Checkpoint -------------
    os.makedirs(cfg.save_ckpt_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.save_ckpt_path, "model.pth"))

    # ------------- Testing -------------
    # Reconstruction test
    os.makedirs(cfg.save_image_path, exist_ok=True)
    samples = train_dataset[:cfg.image_per_row**2]["images"]
    reconsts = model.reconst(samples)
    
    reconsts_grid = make_grid(reconsts.cpu(), cfg.image_per_row)
    reconsts_grid.save(os.path.join(cfg.save_image_path, "reconst.png"))

    # Sample generation
    generator = torch.Generator(device)
    generator.manual_seed(cfg.seed)
    samples = model.sample(cfg.image_per_row**2, generator=generator)

    samples_grid = make_grid(samples.cpu(), cfg.image_per_row)
    samples_grid.save(os.path.join(cfg.save_image_path, "samples.png"))

if __name__ == "__main__":
    main()
