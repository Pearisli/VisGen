import os
import argparse

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from tqdm.auto import tqdm

class ImageFolderDataset(Dataset):

    def __init__(self, directory: str, transform):
        self.directory = directory
        self.files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg')])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        filename = self.files[idx]
        path = os.path.join(self.directory, filename)
        img = Image.open(path).convert("RGB")
        return self.transform(img)

def load_vae(model_name: str, device: torch.device) -> AutoencoderKL:
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        model_name, subfolder="vae", local_files_only=True
    )
    vae.to(device).eval().requires_grad_(False)
    return vae

def build_dataloader(
    input_dir: str,
    resolution: int,
    batch_size: int = 1,
    num_workers: int = 0
) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = ImageFolderDataset(input_dir, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )

@torch.no_grad()
def encode_and_save(
    vae: AutoencoderKL,
    dataloader: DataLoader,
    output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, images in tqdm(enumerate(dataloader), total=len(dataloader), desc="Encoding Images"):
        images = images.to(vae.device)
        latents = vae.encode(images).latent_dist
        params = torch.cat([latents.mean, latents.std], dim=1)
        params = params.squeeze(0).cpu().numpy()

        filename = f"{idx:07d}.npy"
        np.save(os.path.join(output_dir, filename), params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute VAE latent vectors for a folder of images."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory with input JPEG images."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where .npy files will be saved."
    )
    parser.add_argument(
        "--vae_model",
        default="stabilityai/stable-diffusion-2",
        help="Hugging Face checkpoint for the VAE."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Image resolution (square)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers."
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device string ('cuda:0', 'cpu')."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device_str = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    device = torch.device(device_str)

    vae = load_vae(args.vae_model, device)
    loader = build_dataloader(
        input_dir=args.input_dir,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    encode_and_save(vae, loader, args.output_dir)


if __name__ == "__main__":
    main()