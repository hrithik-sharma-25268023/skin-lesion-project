"""Image preprocessing and augmentation pipelines for skin lesion classification."""

from PIL import Image
import torch
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def training_data_transforms() -> transforms.Compose:
    """
    Training image preprocessing and augmentation.
    """

    return transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.90, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomPerspective(distortion_scale=0.10, p=0.20,),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.03),
        transforms.RandomGrayscale(p=0.02), transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def test_val_transforms() -> transforms.Compose:
    """
    Validation/Test preprocessing.
    """

    return transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=IMAGENET_MEAN,
                                                    std=IMAGENET_STD)])


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Converts a normalized tensor back to a PIL image.
    """

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    image_tensor = image_tensor.detach().cpu()
    image_tensor = image_tensor * std + mean
    image_tensor = image_tensor.clamp(0, 1)

    return transforms.ToPILImage()(image_tensor)
