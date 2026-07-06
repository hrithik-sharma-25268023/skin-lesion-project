"""
PyTorch dataset and dataloader utilities for skin lesion classification.
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class SkinLesionDataset(Dataset):
    """
    PyTorch dataset for skin lesion classification.

    Parameters
    ----------
    image_dir : str
        Directory containing the images.

    label_file : str
        Pickle file containing a dictionary
        {image_name: class_index}.

    transform : callable
        Image transformation pipeline.
    """

    def __init__(self, image_dir: str, label_file: str, transform: Callable | None = None) -> None:

        self.image_dir = Path(image_dir)
        self.transform = transform
        with open(label_file, "rb") as file:
            self.labels = pickle.load(file)
        self.images = sorted(self.labels.keys())

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):

        image_name = self.images[index]
        image = Image.open(self.image_dir / image_name).convert("RGB")
        label = self.labels[image_name]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def compute_class_weights(dataset: SkinLesionDataset) -> torch.Tensor:
    """
    Computes inverse-frequency class weights.

    Parameters
    ----------
    dataset : SkinLesionDataset

    Returns
    -------
    torch.Tensor
    """

    labels = list(dataset.labels.values())
    counts = Counter(labels)
    num_classes = len(counts)
    total = len(labels)
    weights = torch.tensor([total / (num_classes * counts[i]) for i in range(num_classes)], dtype=torch.float32)
    return weights


def create_weighted_sampler(dataset: SkinLesionDataset, class_weights: torch.Tensor) -> WeightedRandomSampler:
    """
    Creates a weighted random sampler.

    Parameters
    ----------
    dataset : SkinLesionDataset

    class_weights : torch.Tensor

    Returns
    -------
    WeightedRandomSampler
    """

    sample_weights = [class_weights[label].item() for label in dataset.labels.values()]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def create_dataloader(dataset: SkinLesionDataset, batch_size: int = 32, shuffle: bool = False, sampler=None, num_workers: int = 4) -> DataLoader:
    """
    Creates a PyTorch DataLoader.
    """

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)


def create_dataloaders(train_image_dir: str, train_label_file: str,
                       val_image_dir: str, val_label_file: str, train_transform,
                       val_transform, batch_size: int = 32, num_workers: int = 4):
    """
    Creates train and validation dataloaders.

    Returns
    -------
    tuple
        (
            train_loader,
            val_loader,
            class_weights,
        )
    """

    train_dataset = SkinLesionDataset(image_dir=train_image_dir, label_file=train_label_file, transform=train_transform)
    val_dataset = SkinLesionDataset(image_dir=val_image_dir, label_file=val_label_file, transform=val_transform)
    class_weights = compute_class_weights(train_dataset)
    sampler = create_weighted_sampler(train_dataset, class_weights)
    train_loader = create_dataloader(dataset=train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = create_dataloader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return (train_loader, val_loader, class_weights)
