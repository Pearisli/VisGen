import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from tqdm.auto import tqdm

from models import UNetModel, FlowMatchScheduler, FlowMatchPipeline
from src.dataset.animeface import load_animeface_dataset
from src.util.setting import seed_all, save_checkpoint
from src.util.image_util import generate_and_save_samples
from src.util.lr_scheduler import get_cosine_schedule_with_warmup

class FlowMatchTrainer:

    def __init__(
        self,
        model: UNetModel,
        noise_scheduler: FlowMatchScheduler,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler,
        device: torch.device
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device

    def train(self, dataloader: DataLoader, num_epochs: int):

        self.model.train()
        self.model.to(self.device)

        for epoch in range(num_epochs):
            
            # training
            progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", total=len(dataloader))

            for images, _ in progress_bar:
                
                images: torch.Tensor = images.to(self.device)

                # noise
                noise = torch.randn_like(images)

                # timestep
                batch_size = images.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device="cpu")
                u = F.sigmoid(u)
                indices = (u * self.noise_scheduler.num_train_timesteps).long()
                timesteps = self.noise_scheduler.timesteps[indices].to(self.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = self.noise_scheduler.get_sigmas(timesteps, self.device)
                noisy_images = (1.0 - sigmas) * images + sigmas * noise

                model_pred = self.model(noisy_images, timesteps.long())

                # Flow matching loss
                target = noise - images
                
                loss = F.mse_loss(model_pred, target)
                loss.backward()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.lr_scheduler.get_last_lr()[0]:.6f}"
                })

def main():
    # ------------- Argument & Config -------------
    parser = argparse.ArgumentParser(description="Train Variational Autoencoder")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_flowmatch.yaml",
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
    block_out_channels = tuple([
        hidden_channels, hidden_channels * 2, hidden_channels * 4, hidden_channels * 4
    ])

    model = UNetModel(
        in_channels=3,
        out_channels=3,
        block_out_channels=block_out_channels,
        dropout=cfg.dropout
    )

    noise_scheduler = FlowMatchScheduler(num_train_timesteps=cfg.num_train_timesteps)

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
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
    )
    num_warmup_steps = len(train_dataloader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=cfg.num_epochs * num_warmup_steps
    )

    # ------------- Training -------------
    trainer = FlowMatchTrainer(
        model=model,
        noise_scheduler=noise_scheduler,
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

    pipeline = FlowMatchPipeline(
        sample_size=cfg.resolution,
        unet=model,
        scheduler=noise_scheduler,
    )

    # Sample generation
    generate_and_save_samples(
        model=pipeline,
        device=device,
        save_path=os.path.join(cfg.image_save_dir, "sample.png"),
        seed=cfg.seed,
        normalize=cfg.normalize,
        nrow=cfg.nrow
    )

if __name__ == "__main__":
    main()