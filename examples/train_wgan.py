import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from tqdm.auto import tqdm

from models import WGAN
from src.dataset.animeface import load_animeface_dataset
from src.util.setting import seed_all, save_checkpoint
from src.util.image_util import generate_and_save_samples

class WGANTrainer:

    def __init__(
        self,
        model: WGAN,
        optimizer_critic: optim.Optimizer,
        optimizer_gen: optim.Optimizer,
        device: torch.device,
        lambda_gp: float = 10,
    ):
        self.model = model
        self.optimizer_critic = optimizer_critic
        self.optimizer_gen = optimizer_gen
        self.device = device
        self.lambda_gp = lambda_gp

    def gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        BATCH_SIZE, C, H, W = real.shape
        beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
        interpolated_images = real * beta + fake.detach() * (1 - beta)
        interpolated_images.requires_grad_(True)

        # Calculate critic scores
        mixed_scores = self.model.critic(interpolated_images)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty
    
    def train(self, dataloader: DataLoader, num_epochs: int):

        device = self.device
        self.model.train()
        self.model.to(device)

        for epoch in range(num_epochs):
            
            # training
            progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", total=len(dataloader))

            for i, (real, _) in enumerate(progress_bar):
                real: torch.Tensor = real.to(self.device)
                batch_size: int = real.shape[0]

                noise = self.model.get_noise(batch_size, device)
                fake: torch.Tensor = self.model.generator(noise)

                critic_fake = torch.mean(self.model.critic(fake.detach()))
                critic_real = torch.mean(self.model.critic(real))
                gp = self.gradient_penalty(real, fake, device=device)

                loss_critic = - (critic_real - critic_fake) + self.lambda_gp * gp

                self.optimizer_critic.zero_grad()
                loss_critic.backward()
                self.optimizer_critic.step()

                if i % 5 == 0:
                    gen_fake = self.model.critic(fake)
                    loss_gen = -torch.mean(gen_fake)

                    self.optimizer_gen.zero_grad()
                    loss_gen.backward()
                    self.optimizer_gen.step()
                
                # Clear gradient
                
                progress_bar.set_postfix({
                    "loss_critic": f"{loss_critic.item():.4f}",
                    "loss_gen": f"{loss_gen.item():.4f}",
                })

def main():
    # ------------- Argument & Config -------------
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_wgan.yaml",
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

    model = WGAN(
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
    optimizer_critic = optim.Adam(
        model.critic.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2)
    )
    optimizer_gen = optim.Adam(
        model.generator.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2)
    )

    # ------------- Training -------------
    trainer = WGANTrainer(
        model=model,
        optimizer_critic=optimizer_critic,
        optimizer_gen=optimizer_gen,
        device=device
    )
    trainer.train(train_dataloader, cfg.num_epochs)

    # ------------- Checkpoint Saving -------------
    checkpoint_path = os.path.join(cfg.checkpoint_save_dir, "model.pth")
    save_checkpoint(model, checkpoint_path)

    # ------------- Testing: Reconstruction & Sampling -------------
    os.makedirs(cfg.image_save_dir, exist_ok=True)
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
