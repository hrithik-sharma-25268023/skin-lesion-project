"""Image preprocessing and augmentation pipelines for skin lesion classification."""

from PIL import Image
import torch
from torchvision import transforms

def training_data_transforms() -> transforms.transforms.Compose:
    """training transforms"""

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return train_transform


def test_val_transforms() -> transforms.transforms.Compose:
    """test transforms"""

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    return eval_transform


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Converts a normalized image tensor back to a PIL image for visualization."""

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image_tensor = image_tensor.detach().cpu()
    image_tensor = image_tensor * std + mean
    image_tensor = image_tensor.clamp(0, 1)

    return transforms.ToPILImage()(image_tensor)
