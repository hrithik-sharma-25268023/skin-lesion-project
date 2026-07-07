"""
PyTorch Dataset and DataLoader utilities for Skin Lesion Classification.
"""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from collections import Counter
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
    Subset,
)

# ============================================================
# Label Mapping
# ============================================================

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


# ============================================================
# Dataset
# ============================================================

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

        self.labels = {}

        for image, label in raw_labels.items():

            # Handle test pickle without extension
            if not image.lower().endswith(".jpg"):
                image += ".jpg"

            self.labels[image] = CLASS_TO_IDX[label]

        self.images = sorted(self.labels.keys())

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]

        image_path = self.image_dir / image_name

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        label = self.labels[image_name]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# Class Weights
# ============================================================

def compute_class_weights(dataset):

    if isinstance(dataset, Subset):

        labels = [
            dataset.dataset.labels[
                dataset.dataset.images[idx]
            ]
            for idx in dataset.indices
        ]

    else:

        labels = list(dataset.labels.values())

    counts = Counter(labels)

    total = len(labels)

    num_classes = len(CLASS_TO_IDX)

    weights = torch.zeros(
        num_classes,
        dtype=torch.float32,
    )

    for cls in range(num_classes):

        if counts.get(cls, 0) > 0:

            weights[cls] = (
                total /
                (num_classes * counts[cls])
            )

        else:

            print(
                f"Warning: {IDX_TO_CLASS[cls]} "
                f"not present in current subset."
            )

            weights[cls] = 0.0

    return weights


# ============================================================
# Weighted Sampler
# ============================================================

def create_weighted_sampler(
    dataset,
    class_weights,
):

    if isinstance(dataset, Subset):

        sample_weights = [

            class_weights[
                dataset.dataset.labels[
                    dataset.dataset.images[idx]
                ]
            ].item()

            for idx in dataset.indices

        ]

    else:

        sample_weights = [

            class_weights[label].item()

            for label in dataset.labels.values()

        ]

    return WeightedRandomSampler(

        sample_weights,

        num_samples=len(sample_weights),

        replacement=True,
    )


# ============================================================
# DataLoader
# ============================================================

def create_dataloader(
    dataset,
    batch_size=8,
    shuffle=False,
    sampler=None,
):

    workers = min(4, os.cpu_count())

    return DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle if sampler is None else False,

        sampler=sampler,

        num_workers=workers,

        pin_memory=torch.cuda.is_available(),

        persistent_workers=workers > 0,

        prefetch_factor=2 if workers > 0 else None,
    )


# ============================================================
# Factory
# ============================================================

def create_dataloaders(

    train_image_dir,

    train_label_file,

    val_image_dir,

    val_label_file,

    train_transform,

    val_transform,

    batch_size=8,

    subset_fraction=1.0,
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

    # --------------------------------------------------------

    # Random subset for debugging

    # --------------------------------------------------------

    if subset_fraction < 1.0:

        subset_size = int(
            len(train_dataset) * subset_fraction
        )

        random.seed(42)

        indices = random.sample(
            range(len(train_dataset)),
            subset_size,
        )

        train_dataset = Subset(
            train_dataset,
            indices,
        )

        print(
            f"\nUsing {subset_size} training images "
            f"({subset_fraction*100:.0f}% of dataset)"
        )

    # --------------------------------------------------------

    class_weights = compute_class_weights(
        train_dataset,
    )

    sampler = create_weighted_sampler(
        train_dataset,
        class_weights,
    )

    train_loader = create_dataloader(

        train_dataset,

        batch_size=batch_size,

        sampler=sampler,
    )

    val_loader = create_dataloader(

        val_dataset,

        batch_size=batch_size,

        shuffle=False,
    )

    print(f"Training Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(val_dataset)}")
    print(f"Batch Size        : {batch_size}")
    print(f"Workers           : {min(4, os.cpu_count())}")

    return (
        train_loader,
        val_loader,
        class_weights,
    )
