import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from typing import Optional

def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize images from [-1, 1] to [0, 1] range.
    """
    images = torch.clamp(images, -1.0, 1.0)
    return images * 0.5 + 0.5

def plot_batch(ax: plt.Axes, batch: torch.Tensor, title: Optional[str] = None, **kwargs):
    """
    Plot a batch of images on a given Axes.
    """
    grid_img = make_grid(batch, padding=2, normalize=False).permute(1, 2, 0)
    ax.imshow(grid_img, **kwargs)
    ax.set_axis_off()
    if title:
        ax.set_title(title)

def show_images(
    batch: torch.Tensor,
    title: str = "Images",
    nrow: int = 8,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
    normalize: bool = False,
):
    """
    Display a single batch of images.

    Args:
        batch: Tensor of shape (B, C, H, W)
        title: Title for the image batch
        nrow: Number of images per row
        figsize: Size of the figure (optional)
        save_path: If provided, saves the figure to this path
        normalize: Whether to normalize the images from [-1, 1] to [0, 1]
    """
    if normalize:
        batch = normalize_images(batch)

    if figsize is None:
        figsize = (nrow, nrow)

    fig, ax = plt.subplots(figsize=figsize)
    plot_batch(ax, batch, title)
    plt.show()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

def show_2_batches(
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    title1: str = "Batch 1",
    title2: str = "Batch 2",
    nrow: int = 8,
    figsize: Optional[tuple] = None,
    save_path: Optional[str] = None,
    normalize: bool = False,
):
    """
    Display two batches of images side by side.

    Args:
        batch1: First batch of images
        batch2: Second batch of images
        title1: Title for first batch
        title2: Title for second batch
        nrow: Number of images per row
        figsize: Tuple for figure size
        save_path: Optional path to save the image
        normalize: Whether to normalize the images
    """
    if normalize:
        batch1 = normalize_images(batch1)
        batch2 = normalize_images(batch2)

    if figsize is None:
        figsize = (nrow * 2, nrow)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_batch(axes[0], batch1, title1)
    plot_batch(axes[1], batch2, title2)
    plt.show()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

def show_reconstruction(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    save_path: Optional[str] = None,
    normalize: bool = False,
    nrow: int = 8,
):
    """
    Display original and reconstructed images side-by-side.
    """
    show_2_batches(
        batch1=originals,
        batch2=reconstructions,
        title1="Original Images",
        title2="Reconstructed Images",
        save_path=save_path,
        normalize=normalize,
        nrow=nrow
    )

def show_samples(
    samples: torch.Tensor,
    save_path: Optional[str] = None,
    normalize: bool = False,
    nrow: int = 8,
):
    """
    Display generated sample images.
    """
    show_images(
        batch=samples,
        title="Sample Images",
        save_path=save_path,
        normalize=normalize,
        nrow=nrow
    )

def run_reconstruction_test(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: str,
    normalize: bool = False,
    nrow: int = 8,
):
    """
    Run reconstruction visualization for a batch from the dataloader.

    Args:
        model: Trained model with a `.forward()` returning object with `.sample`.
        dataloader: DataLoader yielding (image, label) batches.
        device: Device to run the model on.
        save_path: Path to save the output image.
        normalize: Whether to normalize images for display.
        nrow: Grid row count.
    """
    model.eval()
    num_samples = nrow * nrow

    images = next(iter(dataloader))[0][:num_samples]
    reconstructions = model.reconstruct(images.to(device)).cpu()

    show_reconstruction(
        originals=images,
        reconstructions=reconstructions,
        save_path=save_path,
        normalize=normalize,
        nrow=nrow
    )

def generate_and_save_samples(
    model: torch.nn.Module,
    device: torch.device,
    save_path: str,
    seed: Optional[int] = 42,
    normalize: bool = False,
    nrow: int = 8,
):
    """
    Generate samples from the model and save the image.

    Args:
        model: The model with a `.sample(n, device, generator)` method.
        n_samples: Number of samples to generate.
        device: Device to use.
        save_path: Path to save the image.
        seed: Manual seed for reproducibility.
        normalize: Whether to normalize the images for display.
        nrow: Grid row count.
    """
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    model.eval()
    samples = model.sample(nrow * nrow, device, generator=generator).cpu()

    show_samples(
        samples=samples,
        save_path=save_path,
        normalize=normalize,
        nrow=nrow
    )