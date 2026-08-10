"""
Image transforms for Swin Transformer training on skin lesion images.

Design notes
------------
- Skin lesions have no canonical orientation, so horizontal AND vertical
  flips are both valid (unlike natural-image datasets).
- RandomResizedCrop + RandAugment + RandomErasing mirror the augmentation
  recipe Swin/ViT-style models were pretrained with, which tends to matter
  more for transformers than for CNNs (they have less built-in inductive
  bias, so they lean more heavily on augmentation to generalize well).
- Color jitter / hue is kept mild since skin tone and lesion coloration
  (e.g. pigmentation) can itself be diagnostically relevant.
"""
from __future__ import annotations

from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def training_data_transforms(image_size: int = 224, randaugment_magnitude: int = 9):
    """
    Training-time augmentation pipeline.

    Parameters
    ----------
    image_size : int
        Final square crop size fed to the model.
    randaugment_magnitude : int
        Strength of RandAugment (0-30). 9 is a moderate setting that works
        well for fine-tuning (vs. training from scratch, which usually
        wants something stronger).
    """
    return transforms.Compose([
        # Scale/translation augmentation - matches the pretraining recipe
        # Swin was trained with, and is gentler than a hard resize+rotate
        # only pipeline.
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(20, interpolation=InterpolationMode.BICUBIC),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        ),
        # RandAugment gives a broader, randomly-sampled set of
        # augmentations (shear, posterize, sharpness, etc.) which is what
        # the original Swin training recipe uses.
        transforms.RandAugment(num_ops=2, magnitude=randaugment_magnitude),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # Random Erasing (applied post-normalization, standard practice)
        # helps prevent over-reliance on any single localized region.
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])


def test_val_transforms(image_size: int = 224):
    """
    Deterministic evaluation pipeline: resize slightly larger than the
    crop size, then center-crop, matching standard ImageNet-style eval
    (keeps train/eval preprocessing consistent in scale).
    """
    resize_size = int(round(image_size / 0.875))  # e.g. 224 -> 256
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
