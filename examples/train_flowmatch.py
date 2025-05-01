import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from tqdm.auto import tqdm
from datasets import load_dataset

from models import UNetModel, FlowMatchScheduler, FlowMatchPipeline
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

def main():
    # ------------- Config -------------
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    # ------------- Seed -------------
    seed_all(cfg.seed)

    # ------------- Model -------------
    model = UNetModel(
        in_channels=3,
        out_channels=3,
        block_out_channels=tuple(cfg.block_out_channels),
        layers_per_block=1
    )

    noise_scheduler = FlowMatchScheduler(num_train_timesteps=cfg.num_train_timesteps)

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
        pin_memory=True,
        persistent_workers=True
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
            
            images: torch.Tensor = batch["images"].to(device)

            # noise
            noise = torch.randn_like(images)

            # timestep
            batch_size = images.shape[0]

            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = torch.normal(mean=0.0, std=1.0, size=(batch_size,), device="cpu")
            u = F.sigmoid(u)
            indices = (u * noise_scheduler.num_train_timesteps).long()
            timesteps = noise_scheduler.timesteps[indices].to(device)

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = noise_scheduler.get_sigmas(timesteps, device)
            noisy_images = (1.0 - sigmas) * images + sigmas * noise

            model_pred = model(noisy_images, timesteps.long())

            # Flow matching loss
            target = noise - images
            
            loss = F.mse_loss(model_pred, target)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.6f}"
            })

    # ------------- Checkpoint -------------
    os.makedirs(cfg.save_ckpt_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.save_ckpt_path, "model.pth"))

    # ------------- Testing -------------
    os.makedirs(cfg.save_image_path, exist_ok=True)
    pipeline = FlowMatchPipeline(
        sample_size=cfg.image_size,
        unet=model,
        scheduler=noise_scheduler,
    )

    # Sample generation
    generator = torch.Generator(device)
    generator.manual_seed(cfg.seed)
    samples = pipeline.sample(cfg.image_per_row**2, generator=generator)

    samples_grid = make_grid(samples.cpu(), cfg.image_per_row)
    samples_grid.save(os.path.join(cfg.save_image_path, "samples.png"))
if __name__ == "__main__":
    main()