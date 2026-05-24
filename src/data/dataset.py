from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class MaskedCelebADataset(Dataset):
    """
    Loads CelebA faces and irregular masks. Returns masked image input (RGB + mask channel)
    and the clean ground-truth image.
    """

    def __init__(
        self,
        image_root: str | Path,
        mask_root: str | Path,
        image_size: int = 256,
        augment: bool = True,
    ) -> None:
        self.image_paths = _scan(image_root, ("*.jpg", "*.png", "*.jpeg"))
        self.mask_paths = _scan(mask_root, ("*.png", "*.jpg"))
        if not self.image_paths:
            raise ValueError(f"No images found under {image_root}")
        if not self.mask_paths:
            raise ValueError(f"No masks found under {mask_root}")

        self.image_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )
        self.mask_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )
        self.augment = augment
        self.hflip = transforms.RandomHorizontalFlip(p=0.5)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask_path = self.mask_paths[idx % len(self.mask_paths)]
        mask = Image.open(mask_path).convert("L")

        if self.augment and random.random() < 0.5:
            img = self.hflip(img)
            mask = self.hflip(mask)

        img_t = self.image_tf(img)
        mask_t = self.mask_tf(mask)
        mask_bin = (mask_t > 0.5).float()

        # input is masked image concatenated with mask channel
        masked_img = img_t * (1.0 - mask_bin)
        model_in = torch.cat([masked_img, mask_bin], dim=0)

        return {"input": model_in, "target": img_t, "mask": mask_bin}


def create_splits(
    dataset: Dataset,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    val_len = max(1, int(len(dataset) * val_ratio))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, lengths=[train_len, val_len], generator=generator)


def _scan(root: str | Path, patterns: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(Path(root).glob(pat)))
    return files

