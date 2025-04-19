import os
import argparse

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as tranforms
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
        return self.transform(img), int(os.path.splitext(filename)[0])

def load_vae(pretrained_model_name_or_path: str, device: torch.device) -> AutoencoderKL:
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    vae.to(device).eval().requires_grad_(False)
    return vae

def build_dataloader(
    input_dir: str,
    resolution: int,
    batch_size: int,
    num_workers: int
) -> DataLoader:
    transform = tranforms.Compose([
        tranforms.Resize((resolution, resolution), interpolation=tranforms.InterpolationMode.BILINEAR),
        tranforms.ToTensor(),
        tranforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = ImageFolderDataset(input_dir, transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

@torch.no_grad()
def encode_and_save(
    vae: AutoencoderKL,
    dataloader: DataLoader,
    output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    for image, name in tqdm(dataloader, desc="Encoding images"):
        image = image.to(vae.device)
        latent = vae.encode(name, return_dict=False)[0]

        params = torch.cat([latent.mean, latent.std], dim=1)
        params = params.squeeze(0).cpu().numpy()

        filename = f"{name.item():07d}.npy"
        np.save(os.path.join(output_dir, filename), params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute VAE latent vectors for a folder of images.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/stable-diffusion-2")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    device_str = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    vae = load_vae(args.pretrained_model_name_or_path, device)
    loader = build_dataloader(
        input_dir=args.input_dir,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    encode_and_save(vae, loader, args.output_dir)

if __name__ == "__main__":
    main()