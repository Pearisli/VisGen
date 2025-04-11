import os
import torch

import random
import numpy as np

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_seed_sequence(
    initial_seed: int,
    length: int,
    min_val=-0x8000_0000_0000_0000,
    max_val=0xFFFF_FFFF_FFFF_FFFF,
):
    if initial_seed is None:
        print("initial_seed is None, reproducibility is not guaranteed")
    random.seed(initial_seed)

    seed_sequence = []

    for _ in range(length):
        seed = random.randint(min_val, max_val)

        seed_sequence.append(seed)

    return seed_sequence

def save_checkpoint(model: torch.nn.Module, save_path: str, state_dict_only: bool = True):
    """
    Save the model checkpoint to a given path.

    Args:
        model: The PyTorch model to save.
        save_path: The full path to save the checkpoint file (e.g., 'checkpoints/model.pth').
        state_dict_only: If True, save only model.state_dict().
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if state_dict_only:
        torch.save(model.state_dict(), save_path)
    else:
        torch.save(model, save_path)
    print(f"Checkpoint saved at {save_path}")