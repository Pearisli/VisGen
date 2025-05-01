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