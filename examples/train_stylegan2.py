import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from math import sqrt

from omegaconf import OmegaConf
from tqdm.auto import tqdm

from models import StyleGAN2
from src.dataset.animeface import load_animeface_dataset
from src.util.setting import seed_all, save_checkpoint
from src.util.image_util import generate_and_save_samples

class PathLengthPenalty(nn.Module):

    def __init__(self, beta: float):

        super().__init__()

        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)

        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        device = x.device
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)

        output = (x * y).sum() / sqrt(image_size)
        sqrt(image_size)

        gradients, *_ = torch.autograd.grad(
            outputs=output,
            inputs=w,
            grad_outputs=torch.ones(output.shape, device=device),
            create_graph=True
        )

        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:

            a = self.exp_sum_a / (1 - self.beta ** self.steps)

            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)

        return loss

class StyleGAN2Trainer:

    def __init__(
        self,
        model: StyleGAN2,
        optimizer_critic: optim.Optimizer,
        optimizer_gen: optim.Optimizer,
        optimizer_map: optim.Optimizer,
        device: torch.device,
        lambda_gp: float = 10,
    ):
        self.model = model
        self.optimizer_critic = optimizer_critic
        self.optimizer_gen = optimizer_gen
        self.optimizer_map = optimizer_map
        self.device = device
        self.lambda_gp = lambda_gp

    def gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor
    ) -> torch.Tensor:
        BATCH_SIZE, C, H, W = real.shape
        beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(self.device)
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

        path_length_penalty = PathLengthPenalty(0.99).to(device)

        for epoch in range(num_epochs):
            
            # training
            progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", total=len(dataloader))

            for i, (real, _) in enumerate(progress_bar):

                real: torch.Tensor = real.to(device)
                batch_size: int = real.shape[0]

                w = self.model.get_w(batch_size, device)

                noise = self.model.get_noise(batch_size, device)

                fake: torch.Tensor = self.model.generator(w, noise)
                critic_fake = self.model.critic(fake.detach())

                critic_real = self.model.critic(real)
                gp = self.gradient_penalty(real, fake)

                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + self.lambda_gp * gp
                    + (0.001 * torch.mean(critic_real ** 2))
                )

                self.optimizer_critic.zero_grad()
                loss_critic.backward()
                self.optimizer_critic.step()

                gen_fake = self.model.critic(fake)
                loss_gen = -torch.mean(gen_fake)

                if i % 16 == 0:
                    plp = path_length_penalty(w, fake)
                    if not torch.isnan(plp):
                        loss_gen = loss_gen + plp
                
                self.optimizer_map.zero_grad()
                self.optimizer_gen.zero_grad()

                loss_gen.backward()
                self.optimizer_gen.step()
                self.optimizer_map.step()
                
                progress_bar.set_postfix({
                    "gp": f"{gp.item():.4f}",
                    "loss_critic": f"{loss_critic.item():.4f}",
                })

def main():
    # ------------- Argument & Config -------------
    parser = argparse.ArgumentParser(description="Train Autoencoder")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_stylegan2.yaml",
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
    model = StyleGAN2(
        log_resolution=cfg.log_resolution,
        latent_channels=cfg.latent_channels,
        w_dim=cfg.w_dim,
        mapping_layers=cfg.mapping_layers,
        n_features=cfg.n_features
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
        betas=(0.0, 0.99)
    )
    optimizer_gen = optim.Adam(
        model.generator.parameters(),
        lr=cfg.learning_rate,
        betas=(0.0, 0.99)
    )
    optimizer_map = optim.Adam(
        model.mapping_network.parameters(),
        lr=cfg.learning_rate,
        betas=(0.0, 0.99)
    )

    # ------------- Training -------------
    trainer = StyleGAN2Trainer(
        model=model,
        optimizer_critic=optimizer_critic,
        optimizer_gen=optimizer_gen,
        optimizer_map=optimizer_map,
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