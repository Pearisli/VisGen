import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA

from dataclasses import dataclass
from tqdm.auto import tqdm
from typing import Tuple

from src.modules.convolution import (
    ConvBlock,
    DownBlock,
    UpBlock,
    StackBlock,
)
from src.modules.vqperceptual import VQLPIPSWithDiscriminator
from src.util.seeding import seed_all
from src.util.image_util import PlotManager

CHECKPOINT_SAVE_DIR = "./checkpoints/vqgan"
IMAGE_SAVE_DIR = "./assets/images/vqgan"

NUM_DOWNSAMPLE = 5
IMGAE_SIZE = 128

class Encoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...] = (64,),
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.block_out_channels = block_out_channels

        self.conv_in = ConvBlock(in_channels, block_out_channels[0])
        self.down_blocks = StackBlock(block_out_channels, DownBlock)
        self.conv_out = ConvBlock(block_out_channels[-1], latent_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(x)
        hidden_states = self.down_blocks(hidden_states)
        z = self.conv_out(hidden_states)
        return z
    
class Decoder(nn.Module):

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...] = (64,),
    ) -> None:
        super().__init__()
        
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.block_out_channels = block_out_channels

        reversed_block_out_channels = list(reversed(block_out_channels))

        self.conv_in = ConvBlock(latent_channels, reversed_block_out_channels[0])
        self.up_blocks = StackBlock(reversed_block_out_channels, UpBlock)
        self.conv_out = nn.Conv2d(
            in_channels=reversed_block_out_channels[-1],
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.nonlinearity = nn.Sigmoid()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(z)
        hidden_states = self.up_blocks(hidden_states)
        x = self.conv_out(hidden_states)
        x = self.nonlinearity(x)
        return x

class VectorQuantizer(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commiment_cost: float = 0.25
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.commiment_cost = commiment_cost
        self._init_parameters()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_q = self.quantize(z)  # Quantize the latents

        commiment_loss = F.mse_loss(z_q.detach(), z)
        embedding_loss = F.mse_loss(z_q, z.detach())
        vq_loss = embedding_loss + self.commiment_cost * commiment_loss

        # Make the gradient with respect to latents be equal to the gradient with respect to quantized latents 
        z_q = z + (z_q - z).detach()
        return z_q, vq_loss
    
    def quantize(self, z: torch.IntTensor) -> torch.Tensor:
        # Compute L2 distances between latents and embedding weights
        dist = torch.linalg.vector_norm(z.movedim(1, -1).unsqueeze(-2) - self.embedding.weight, dim=-1)
        # Get the number of the nearest codebook vector
        encoding_inds = torch.argmin(dist, dim=-1)

        z_q = self.embedding(encoding_inds)
        z_q = z_q.movedim(-1, 1) # Move channels back
        return z_q
    
    def _init_parameters(self):
        nn.init.uniform_(self.embedding.weight, -1 / self.num_embeddings, 1 / self.num_embeddings)

# class Encoder(nn.Module):

#     def __init__(
#         self,
#         in_channels: int = 3,
#         out_channels: int = 3,
#         block_out_channels: Tuple[int, ...] = (64,),
#         layers_per_block: int = 1,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#         self.layers_per_block = layers_per_block

#         self.conv_in = nn.Conv2d(
#             in_channels,
#             block_out_channels[0],
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )

#         self.down_blocks = nn.ModuleList([])

#         # down
#         output_channel = block_out_channels[0]
#         for i, next_output_channel in enumerate(block_out_channels):
#             input_channel = output_channel
#             output_channel = next_output_channel
#             is_final_block = (i == len(block_out_channels) - 1)

#             self.down_blocks.append(DownEncoderBlock(
#                 in_channels=input_channel,
#                 out_channels=output_channel,
#                 dropout=dropout,
#                 num_layers=layers_per_block,
#                 add_downsample=not is_final_block
#             ))
    
#         # out
#         self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=32, eps=1e-6)
#         self.conv_act = nn.SiLU()
#         self.conv_out = nn.Conv2d(block_out_channels[-1], out_channels, 3, padding=1)

#     def forward(self, sample: torch.Tensor) -> torch.Tensor:
#         sample = self.conv_in(sample)

#         for down_block in self.down_blocks:
#             sample = down_block(sample)

#         sample = self.conv_norm_out(sample)
#         sample = self.conv_act(sample)
#         sample = self.conv_out(sample)

#         return sample

# class Decoder(nn.Module):

#     def __init__(
#         self,
#         in_channels: int = 3,
#         out_channels: int = 3,
#         block_out_channels: Tuple[int, ...] = (64,),
#         layers_per_block: int = 1,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#         self.layers_per_block = layers_per_block

#         self.conv_in = nn.Conv2d(
#             in_channels,
#             block_out_channels[-1],
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )

#         self.up_blocks = nn.ModuleList([])

#         # up
#         reversed_block_out_channels = list(reversed(block_out_channels))
#         output_channel = reversed_block_out_channels[0]
#         for i, next_output_channel in enumerate(reversed_block_out_channels):
#             input_channel = output_channel
#             output_channel = next_output_channel
#             is_final_block = (i == len(block_out_channels) - 1)

#             self.up_blocks.append(UpDecoderBlock(
#                 in_channels=input_channel,
#                 out_channels=output_channel,
#                 dropout=dropout,
#                 num_layers=layers_per_block,
#                 add_upsample=not is_final_block
#             ))
    
#         # out
#         self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-6)
#         self.conv_act = nn.SiLU()
#         self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

#     def forward(self, sample: torch.Tensor) -> torch.Tensor:
#         sample = self.conv_in(sample)

#         for up_block in self.up_blocks:
#             sample = up_block(sample)

#         sample = self.conv_norm_out(sample)
#         sample = self.conv_act(sample)
#         sample = self.conv_out(sample)

#         return sample

class VectorQuantizer(nn.Module):

    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = z.permute(0, 2, 3, 1).contiguous() # [B, C, H, W] -> [B, H, W, C]
        z_flattened = z.view(-1, self.e_dim)

        d = (torch.sum(z_flattened ** 2, dim=1, keepdim=True) + 
            torch.sum(self.embedding.weight**2, dim=1) - 2 *
            (z_flattened @ self.embedding.weight.T)
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q: torch.Tensor = self.embedding(min_encoding_indices).view(z.shape)

        commiment_loss = F.mse_loss(z_q.detach(), z)
        embedding_loss = F.mse_loss(z_q, z.detach())
        vq_loss = embedding_loss + self.beta * commiment_loss

        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, vq_loss

@dataclass
class VQGANOutput:

    dec: torch.Tensor
    vq_loss: torch.Tensor

class VQGAN(nn.Module):

    def __init__(
        self,
        sample_size: int,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 32,
        num_embeddings: int = 256,
        block_out_channels: Tuple[int, ...] = (64,),
    ):
        super().__init__()

        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_embeddings = num_embeddings
        self.block_out_channels = block_out_channels

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
        )

        self.vq = VectorQuantizer(num_embeddings, latent_channels)

        self.decoder = Decoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
        )

        # self.encoder = Encoder(
        #     in_channels=in_channels,
        #     out_channels=latent_channels,
        #     block_out_channels=block_out_channels,
        # )

        # self.vq = VectorQuantizer(num_embeddings, latent_channels)

        # self.decoder = Decoder(
        #     in_channels=latent_channels,
        #     out_channels=out_channels,
        #     block_out_channels=block_out_channels,
        # )

        self.quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)
    
    def get_last_layer(self) -> torch.Tensor:
        return self.decoder.conv_out.weight
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.encoder(x)
        hidden_states = self.quant_conv(hidden_states)
        quant, vq_loss = self.vq(hidden_states)
        return quant, vq_loss
    
    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    
    def forward(self, x: torch.Tensor) -> VQGANOutput:
        quant, vq_loss = self.encode(x)
        dec = self.decode(quant)
        return VQGANOutput(
            dec=dec,
            vq_loss=vq_loss
        )

class VQGANTrainer:

    def __init__(
        self,
        model: VQGAN,
        loss: VQLPIPSWithDiscriminator,
        optimizer_ae: optim.Optimizer,
        optimizer_disc: optim.Optimizer,
        device: torch.device,
    ):
        self.model = model
        self.loss = loss
        self.optimizer_ae = optimizer_ae
        self.optimizer_disc = optimizer_disc
        self.device = device
    
    def train(self, dataloader: DataLoader, num_epochs: int):

        self.model.train()
        self.model.to(self.device)
        self.loss.to(self.device)
        global_step = 0

        for i in range(num_epochs):
            
            # training
            train_tbar = tqdm(
                dataloader,
                desc=f"Epoch [{i + 1}/{num_epochs}]",
                total=len(dataloader)
            )

            for real, _ in train_tbar:
                real: torch.Tensor = real.to(self.device)
                output: VQGANOutput = self.model(real)
                
                aeloss = self.loss.ae_loss(
                    vq_loss=output.vq_loss,
                    inputs=real,
                    reconstructions=output.dec,
                    global_step=global_step,
                    last_layer=self.model.get_last_layer()
                )
                aeloss.backward()
                self.optimizer_ae.step()
                self.optimizer_ae.zero_grad()

                disc_loss = None
                if global_step >= self.loss.discriminator_iter_start:
                    disc_loss = self.loss.disc_loss(
                        inputs=real,
                        reconstructions=output.dec,
                    )
                    disc_loss.backward()
                    self.optimizer_disc.step()
                    self.optimizer_disc.zero_grad()
                
                train_tbar.set_postfix({
                    "ae_loss": f"{aeloss.item():.4f}",
                    "disc_loss": f"{(0 if disc_loss is None else disc_loss.item()):.4f}",
                })

                global_step += 1

def parse_args():
    parser = argparse.ArgumentParser(description="Train a VQVAE with configurable parameters.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproductivity")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for data loading")
    parser.add_argument("--num_epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--latent_channels", type=int, default=32, help="Number of latent channels in the model")
    parser.add_argument("--hidden_channels", type=int, default=64, help="Hidden channel size (first layer of block_out_channels)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()
    
def main():
    args = parse_args()

    seed_all(args.seed)

    #model
    hidden_channels = args.hidden_channels
    block_out_channels = tuple([hidden_channels * 2**i for i in range(NUM_DOWNSAMPLE + 1)])

    model = VQGAN(
        sample_size=IMGAE_SIZE // (2 ** NUM_DOWNSAMPLE),
        in_channels=3,
        out_channels=3,
        latent_channels=args.latent_channels,
        block_out_channels=block_out_channels,
    )
    loss = VQLPIPSWithDiscriminator(disc_start=25000)

    # data
    train_transform = transforms.Compose([
        transforms.Resize(IMGAE_SIZE),
        transforms.CenterCrop(IMGAE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CelebA("./data", transform=train_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # train
    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    
    optimizer_ae = optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.5, 0.9)
    )
    optimizer_disc = optim.Adam(
        loss.discriminator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.9)
    )

    trainer = VQGANTrainer(
        model=model,
        loss=loss,
        optimizer_ae=optimizer_ae,
        optimizer_disc=optimizer_disc,
        device=device
    )
    trainer.train(train_dataloader, args.num_epochs)

    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_SAVE_DIR, exist_ok=True)

    # save checkpoint
    checkpoint_save_path = os.path.join(CHECKPOINT_SAVE_DIR, "model.pth")
    print(f"Save checkpoint at {checkpoint_save_path}")
    torch.save(model, checkpoint_save_path)
    
    # test
    model.eval()
    n_samples = 64
    images = next(iter(train_dataloader))[0][:n_samples]

    # reconstruction
    with torch.no_grad():
        reconst_images = model(images.to(device)).dec
        reconst_images = (reconst_images + 1.0) / 2
        PlotManager.show_2_batches(images, reconst_images.cpu(), "Original Images", "Reconstructed Images")
        PlotManager.save(os.path.join(IMAGE_SAVE_DIR, "reconstruction.png"))

if __name__ == "__main__":
    main()