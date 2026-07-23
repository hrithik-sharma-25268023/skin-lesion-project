"""
PyTorch Dataset and DataLoader utilities for Skin Lesion Classification.
"""

from __future__ import annotations

import io
import os
import pickle
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Callable

import boto3
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset

CLASS_TO_IDX = {"MEL": 0, "NV": 1, "BCC": 2, "AK": 3,
                "BKL": 4, "DF": 5, "VASC": 6, "SCC": 7}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


def downsample_dataset(dataset, max_samples_per_class, random_seed=42):
    """
    Downsample majority classes while keeping minority classes unchanged.

    Parameters
    ----------
    dataset : SkinLesionDataset
    max_samples_per_class : dict
    """

    random.seed(random_seed)
    class_indices = defaultdict(list)

    for idx, image_name in enumerate(dataset.images):
        label = dataset.labels[image_name]
        class_indices[label].append(idx)
    print("\nOriginal Training Distribution")
    for cls in sorted(class_indices):
        print(f"{IDX_TO_CLASS[cls]:<5}: {len(class_indices[cls])}")
    selected_indices = []
    for cls, indices in class_indices.items():
        class_name = IDX_TO_CLASS[cls]
        max_keep = max_samples_per_class.get(class_name, len(indices))
        if len(indices) > max_keep:
            indices = random.sample(indices, max_keep)
        selected_indices.extend(indices)
    random.shuffle(selected_indices)
    print("\nBalanced Training Distribution")
    balanced = defaultdict(int)

    for idx in selected_indices:
        image_name = dataset.images[idx]
        label = dataset.labels[image_name]
        balanced[label] += 1

    for cls in sorted(balanced):
        print(f"{IDX_TO_CLASS[cls]:<5}: {balanced[cls]}")
    print(f"\nTraining Images : {len(selected_indices)}")
    return Subset(dataset, selected_indices)


class SkinLesionDataset(Dataset):

    def __init__(self, storage_type="local", image_dir=None, label_file=None,
                 bucket_name=None, image_prefix=None,
                 label_key=None, transform=None):

        self.storage_type = storage_type
        self.transform = transform
        if storage_type == "local":
            self.image_dir = Path(image_dir)
        else:
            self.bucket = bucket_name
            self.image_prefix = image_prefix
            self.s3 = boto3.client("s3")

        if self.storage_type == "local":
            with open(label_file, "rb") as f:
                raw_labels = pickle.load(f)
        else:
            response = self.s3.get_object(Bucket=self.bucket, Key=label_key)
            raw_labels = pickle.loads(response["Body"].read())

        self.labels = {}

        for image, label in raw_labels.items():
            if not image.lower().endswith(".jpg"):
                image += ".jpg"
            self.labels[image] = CLASS_TO_IDX[label]
        self.images = sorted(self.labels.keys())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_name = self.images[index]
        if self.storage_type == "local":
            image_path = self.image_dir / image_name
            image = Image.open(image_path).convert("RGB")
        else:
            key = f"{self.image_prefix}/{image_name}"
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            image = Image.open(io.BytesIO(response["Body"].read())).convert("RGB")

        label = self.labels[image_name]

        if self.transform:
            image = self.transform(image)
        return image, label


def compute_class_weights(dataset):

    if isinstance(dataset, Subset):
        labels = [dataset.dataset.labels[dataset.dataset.images[idx]] for idx in dataset.indices]
    else:
        labels = list(dataset.labels.values())
    counts = Counter(labels)
    total = len(labels)
    num_classes = len(CLASS_TO_IDX)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls in range(num_classes):
        if counts.get(cls, 0) > 0:
            weights[cls] = (total / (num_classes * counts[cls]))
        else:
            print(f"Warning: {IDX_TO_CLASS[cls]} not present in current subset.")
            weights[cls] = 0.0
    return weights


def create_weighted_sampler(dataset, class_weights):

    if isinstance(dataset, Subset):
        sample_weights = [
            class_weights[dataset.dataset.labels[dataset.dataset.images[idx]]].item()
            for idx in dataset.indices
        ]
    else:
        sample_weights = [class_weights[label].item() for label in dataset.labels.values()]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def create_dataloader(dataset, batch_size=8, shuffle=False, sampler=None):

    workers = min(16, os.cpu_count())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        drop_last=shuffle,  # avoids a lone tiny batch (bad for BatchNorm / MixUp) during training
    )


def create_dataloaders(
    train_transform,
    val_transform,
    batch_size=8,
    subset_fraction=1.0,
    downsample_classes=None,
    use_weighted_sampler=True,
    train_image_dir=None,
    train_label_file=None,
    val_image_dir=None,
    val_label_file=None,
    bucket_name=None,
    train_image_prefix=None,
    train_label_key=None,
    val_image_prefix=None,
    val_label_key=None):

    storage_type = "s3" if bucket_name is not None else "local"
    if storage_type == "local":
        train_dataset = SkinLesionDataset(
            storage_type="local",
            image_dir=train_image_dir,
            label_file=train_label_file,
            transform=train_transform
        )

        val_dataset = SkinLesionDataset(
            storage_type="local",
            image_dir=val_image_dir,
            label_file=val_label_file,
            transform=val_transform
        )

    else:
        train_dataset = SkinLesionDataset(
            storage_type="s3",
            bucket_name=bucket_name,
            image_prefix=train_image_prefix,
            label_key=train_label_key,
            transform=train_transform
        )

        val_dataset = SkinLesionDataset(
            storage_type="s3",
            bucket_name=bucket_name,
            image_prefix=val_image_prefix,
            label_key=val_label_key,
            transform=val_transform)

    if subset_fraction < 1.0:
        subset_size = int(len(train_dataset) * subset_fraction)
        random.seed(42)
        indices = random.sample(range(len(train_dataset)), subset_size)

        train_dataset = Subset(train_dataset, indices)
        print(f"\nUsing {subset_size} training images ({subset_fraction * 100:.0f}% of dataset)")

    if downsample_classes is not None:
        if isinstance(train_dataset, Subset):
            raise ValueError("Use either subset_fraction or downsample_classes, not both.")
        train_dataset = downsample_dataset(train_dataset, downsample_classes)

    class_weights = compute_class_weights(train_dataset)
    if use_weighted_sampler:
        sampler = create_weighted_sampler(train_dataset, class_weights)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train_loader = create_dataloader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle)
    val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Training Images   : {len(train_dataset)}")
    print(f"Validation Images : {len(val_dataset)}")
    print(f"Batch Size        : {batch_size}")
    print(f"Workers           : {min(16, os.cpu_count())}")
    print(f"Weighted Sampler  : {use_weighted_sampler}")
    print(f"Downsampling      : {'Enabled' if downsample_classes is not None else 'Disabled'}")
    return train_loader, val_loader, class_weights
