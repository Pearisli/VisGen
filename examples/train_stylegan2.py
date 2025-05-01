import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from math import sqrt

from omegaconf import OmegaConf
from tqdm.auto import tqdm
from datasets import load_dataset

from models import StyleGAN2
from src.util.setting import seed_all
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

def gradient_penalty(
    critic: torch.nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    device = next(critic.parameters()).device

    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

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

def main():
    # ------------- Config -------------
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    # ------------- Seed -------------
    seed_all(cfg.seed)

    # ------------- Model -------------
    model = StyleGAN2(
        log_resolution=cfg.log_resolution,
        latent_channels=cfg.latent_channels,
        w_dim=cfg.w_dim,
        mapping_layers=cfg.mapping_layers,
        n_features=cfg.n_features
    )

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

    num_epochs = cfg.num_epochs

    # ------------- Training -------------
    model.train()
    model.to(device)

    path_length_penalty = PathLengthPenalty(0.99).to(device)

    for epoch in range(num_epochs):
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch [{epoch+1}/{num_epochs}]",
            total=len(train_dataloader)
        )

        for i, batch in enumerate(progress_bar):

            real: torch.Tensor = batch["images"].to(device)
            batch_size: int = real.shape[0]

            w = model.get_w(batch_size)

            noise = model.get_noise(batch_size)

            fake: torch.Tensor = model.generator(w, noise)
            critic_fake = model.critic(fake.detach())

            critic_real = model.critic(real)
            gp = gradient_penalty(model.critic, real, fake)

            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + cfg.lambda_gp * gp
                + (0.001 * torch.mean(critic_real ** 2))
            )

            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()

            gen_fake = model.critic(fake)
            loss_gen = -torch.mean(gen_fake)

            if i % 16 == 0:
                plp = path_length_penalty(w, fake)
                if not torch.isnan(plp):
                    loss_gen = loss_gen + plp
            
            optimizer_map.zero_grad()
            optimizer_gen.zero_grad()

            loss_gen.backward()
            optimizer_gen.step()
            optimizer_map.step()
            
            progress_bar.set_postfix({
                "gp": f"{gp.item():.4f}",
                "loss_critic": f"{loss_critic.item():.4f}",
            })

    # ------------- Checkpoint Saving -------------
    os.makedirs(cfg.save_ckpt_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.save_ckpt_path, "model.pth"))

    # ------------- Testing -------------
    os.makedirs(cfg.save_image_path, exist_ok=True)
    generator = torch.Generator(device)
    generator.manual_seed(cfg.seed)
    samples = model.sample(cfg.image_per_row**2, generator=generator)

    samples_grid = make_grid(samples.cpu(), cfg.image_per_row)
    samples_grid.save(os.path.join(cfg.save_image_path, "samples.png"))

if __name__ == "__main__":
    main()