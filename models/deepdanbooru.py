import os
import json
from PIL.Image import Image

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_tensor
from typing import List, Tuple, Union
from safetensors.torch import load_file, save_file
from huggingface_hub import PyTorchModelHubMixin

class Bottleneck(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        add_downsample: bool = False
    ) -> None:
        super().__init__()
        width = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if add_downsample:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class DeepDanbooruModel(nn.Module, PyTorchModelHubMixin):

    resolution = 512

    def __init__(
        self,
        block_out_channels: Tuple[int, ...],
        blocks_per_layer: Tuple[int, ...],
        num_classes: int,
    ) -> None:
        super().__init__()
        in_channels = 64

        self.block_out_channels = block_out_channels
        self.blocks_per_layer = blocks_per_layer
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=3)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layers = nn.ModuleList([])
        input_channel = in_channels
        for i, (num_block, output_channel) in enumerate(zip(blocks_per_layer, block_out_channels)):
            stride = 1 if i == 0 else 2
            self.layers.append(
                self._make_layer(num_block, input_channel, output_channel, stride=stride)
            )
            input_channel = output_channel

        self.fc = nn.Conv2d(in_channels=block_out_channels[-1], out_channels=num_classes, kernel_size=1, bias=False)
        self.activation = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                ks = m.kernel_size[0]
                if ks > 1:
                    m.padding_mode = "constant"
                    if m.stride[0] == 1:
                        m._reversed_padding_repeated_twice = (1, 1, 1, 1)
                    else:
                        m._reversed_padding_repeated_twice = (
                            ks // 2 - 1, ks // 2,
                            ks // 2 - 1, ks // 2
                        )

        self._tags = []

    def _make_layer(
        self,
        num_block: int,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> nn.Sequential:

        layers = nn.Sequential()
        layers.append(
            Bottleneck(in_channels, out_channels, stride=stride, add_downsample=True)
        )

        for _ in range(1, num_block):
            layers.append(
                Bottleneck(out_channels, out_channels, stride=1, add_downsample=False)
            )

        return layers
    
    def _save_pretrained(self, save_directory: str, config: dict = None, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        # 1. Save weights
        save_file(self.state_dict(), os.path.join(save_directory, "model.safetensors"))
        # 2. Save config
        if config is None:
            # assume these were passed at init and stored
            config = {
                "block_out_channels": self.block_out_channels,
                "blocks_per_layer": self.blocks_per_layer,
                "num_classes": self.num_classes
            }
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        # 3. Save tags
        if hasattr(self, "_tags") and self._tags:
            with open(os.path.join(save_directory, "tags.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(self._tags))

    @classmethod
    def _from_pretrained(cls, model_id: str, **kwargs) -> "DeepDanbooruModel":
        # Download & extract if needed
        repo_path = model_id
        # Use HF utilities to download: snapshot_download or hf_hub_download under the hood
        # Here we assume local path
        # 1. Load config
        with open(os.path.join(repo_path, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model = cls(tuple(cfg["block_out_channels"]), tuple(cfg["blocks_per_layer"]), cfg["num_classes"])
        # 2. Load weights
        state = load_file(os.path.join(repo_path, "model.safetensors"))
        model.load_state_dict(state, strict=True)
        # 3. Load tags
        tags_file = os.path.join(repo_path, "tags.txt")
        if os.path.isfile(tags_file):
            with open(tags_file, "r", encoding="utf-8") as f:
                model._tags = [line.strip() for line in f if line.strip()]
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for layer in self.layers:
            x = layer(x)

        x = self.fc(x)
        x = nn.functional.avg_pool2d(x, kernel_size=x.shape[-2:])
        
        x = torch.flatten(x, 1)
        x = self.activation(x)
        
        return x
    
    @torch.no_grad()
    def tag(
        self,
        image: Union[Image, List[Image], torch.Tensor],
        threshold: float = 0.5
    ) -> List[List[str]]:
        
        # Convert PIL Images to tensors if needed and stack
        if isinstance(image, Image):
            image = [image, ]
        if isinstance(image, List):
            images = torch.stack([to_tensor(img) for img in image])
        
        assert images.ndim == 4, f"Expected 4D tensor, got shape {images.shape}"
        device = next(self.parameters()).device
        images = images.to(device)

        # Model forward pass
        probs = self(images)

        # Thresholding and tag lookup
        results = []
        for prob_vector in probs:
            selected = (prob_vector > threshold).nonzero(as_tuple=True)[0].cpu().tolist()
            results.append([self._tags[i] for i in selected])

        # Return single result or batch
        return results