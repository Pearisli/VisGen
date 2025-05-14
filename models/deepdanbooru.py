import os
from PIL import Image

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_tensor, resize
from typing import List, Tuple, Union
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

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

    def __init__(
        self,
        block_out_channels: Tuple[int, ...],
        blocks_per_layer: Tuple[int, ...],
        num_classes: int,
        *,
        tag_file: str,
        resolution: int = 512
    ) -> None:
        super().__init__()
        in_channels = 64

        self.block_out_channels = block_out_channels
        self.blocks_per_layer = blocks_per_layer
        self.num_classes = num_classes
        self._tags = []
        with open(tag_file, "r", encoding="utf-8") as f:
            self._tags = [line.strip() for line in f if line.strip()]
        self.resolution = resolution

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
        image: Union[Image.Image, List[Image.Image], torch.Tensor],
        threshold: float = 0.5
    ) -> List[List[str]]:
        
        # Convert PIL Images to tensors if needed and stack
        if isinstance(image, Image.Image):
            image = [image, ]
        if isinstance(image, List):
            images = torch.stack([
                resize(to_tensor(img), [self.resolution, self.resolution])
            for img in image])
        
        assert images.ndim == 4 and images.shape[-1] == self.resolution and images.shape[-2] == self.resolution, f"Expected 4D tensor (N, C, 512, 512), got shape {images.shape}"
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

    def save_pretrained(
        self,
        save_directory: str,
        config: dict = None,
        **kwargs
    ):
        # 1. Call super to save model and config
        super().save_pretrained(save_directory, config=config, **kwargs)

        readme = os.path.join(save_directory, "README.md")
        if os.path.exists(readme):
            os.remove(readme)

        # 2. Write tags.txt into save_directory
        tags_path = os.path.join(save_directory, "tags.txt")
        with open(tags_path, "w", encoding="utf-8") as f:
            for tag in getattr(self, '_tags', []):
                f.write(f"{tag}\n")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        force_download: bool = False,
        cache_dir: str = None,
        local_files_only: bool = False,
        **model_kwargs
    ):
        model_id = str(pretrained_model_name_or_path)
        if os.path.isdir(model_id):
            tag_file = os.path.join(model_id, "tags.txt")
        else:
            tag_file = hf_hub_download(
                repo_id=model_id,
                filename="tags.txt",
                force_download=force_download,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        model_kwargs["tag_file"] = tag_file
        
        instance = super().from_pretrained(
            pretrained_model_name_or_path,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **model_kwargs
        )
        return instance