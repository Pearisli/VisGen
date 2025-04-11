from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def load_animeface_dataset(
    base_data_dir: str,
    resolution: int,
    batch_size: int,
    num_workers: int = 8,
    normalize: bool = False
) -> DataLoader:

    train_transform_list = [
        transforms.Resize((resolution, resolution)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if normalize:
        train_transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    train_transform = transforms.Compose(train_transform_list)

    train_dataset = ImageFolder(base_data_dir, transform=train_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_dataloader