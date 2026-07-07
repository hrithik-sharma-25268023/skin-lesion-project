"""
PyTorch Dataset and DataLoader utilities for skin lesion classification.
"""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

# ---------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------

CLASS_TO_IDX = {
    "MEL": 0,
    "NV": 1,
    "BCC": 2,
    "AK": 3,
    "BKL": 4,
    "DF": 5,
    "VASC": 6,
    "SCC": 7,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------


class SkinLesionDataset(Dataset):

    def __init__(
        self,
        image_dir: str,
        label_file: str,
        transform: Callable | None = None,
    ):

        self.image_dir = Path(image_dir)
        self.transform = transform

        with open(label_file, "rb") as f:
            raw_labels = pickle.load(f)

        # Convert labels ONCE
        self.labels = {
            image: CLASS_TO_IDX[label]
            for image, label in raw_labels.items()
        }

        self.images = sorted(self.labels.keys())

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        image = Image.open(
            self.image_dir / image_name
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[image_name]

        return image, label


# ---------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------


def compute_class_weights(dataset):

    labels = list(dataset.labels.values())

    counts = Counter(labels)

    print("\nCounts:", counts)

    num_classes = len(CLASS_TO_IDX)

    total = len(labels)

    weights = []

    for cls in range(num_classes):

        count = counts.get(cls, 0)

        if count == 0:
            raise ValueError(
                f"Class {cls} has zero samples in the training set."
            )

        weights.append(
            total / (num_classes * count)
        )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    print("\nClass Weights")

    for cls, weight in enumerate(weights):

        print(
            f"{IDX_TO_CLASS[cls]} : {weight:.3f}"
        )

    return weights


# ---------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------


def create_weighted_sampler(
    dataset,
    class_weights,
):

    sample_weights = [
        class_weights[label].item()
        for label in dataset.labels.values()
    ]

    return WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------


def create_dataloader(
    dataset,
    batch_size=32,
    shuffle=False,
    sampler=None,
    num_workers=4,
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler else shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def create_dataloaders(
    train_image_dir,
    train_label_file,
    val_image_dir,
    val_label_file,
    train_transform,
    val_transform,
    batch_size=32,
    num_workers=4,
):

    train_dataset = SkinLesionDataset(
        train_image_dir,
        train_label_file,
        train_transform,
    )

    val_dataset = SkinLesionDataset(
        val_image_dir,
        val_label_file,
        val_transform,
    )

    print("\nExample label:", train_dataset[0][1])

    class_weights = compute_class_weights(
        train_dataset
    )

    sampler = create_weighted_sampler(
        train_dataset,
        class_weights,
    )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_loader,
        val_loader,
        class_weights,
    )