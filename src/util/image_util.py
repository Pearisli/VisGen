from PIL import Image
import numpy as np
import torch
import torchvision.utils as vutils

def normalize(images: torch.Tensor) -> torch.Tensor:
    return 2.0 * images - 1.0

def denormalize(images: torch.Tensor) -> torch.Tensor:
    return (images / 2 + 0.5).clamp(0, 1)

def make_grid(images: torch.Tensor, nrow: int) -> Image.Image:
    images_grid = vutils.make_grid(images, nrow).permute(1, 2, 0).numpy()
    images_grid = (images_grid * 255).astype(np.uint8)
    return Image.fromarray(images_grid)