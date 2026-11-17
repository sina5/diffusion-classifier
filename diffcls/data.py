"""CIFAR-10 as uint8 tensors, and the conversion to the [-1, 1] range that the DiT expects."""

from pathlib import Path

import torch
from torchvision import datasets

from diffcls import constants as C


class Cifar10Split:
    """One CIFAR-10 split held in memory as uint8 images (N, 3, 32, 32) and labels (N,)."""

    def __init__(self, root: Path = C.DATA_DIR, train: bool = False) -> None:
        self.root = root
        self.train = train

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the split (download it to root if it is not there)."""
        raw = datasets.CIFAR10(str(self.root), train=self.train, download=True)
        images = torch.from_numpy(raw.data).permute(0, 3, 1, 2).contiguous()
        return images, torch.tensor(raw.targets, dtype=torch.long)


def to_model_range(images: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] to float [-1, 1]."""
    return images.float() / (C.UINT8_MAX / 2) - 1.0
