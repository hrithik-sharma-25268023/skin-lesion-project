"""test cases for train and eval-test transforms"""

import random
import numpy as np
from PIL import Image, ImageChops
import torch

from skin_lesion_project.processing import image_processing



def test_training_transform_matches_reference():
    """tests the train transforms"""

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    original = Image.open("/mnt/d/skin-lesion-data/2019+2020/train/ISIC_0000013_2019.jpg").convert("RGB")
    transform = image_processing.training_data_transforms()
    transformed = image_processing.tensor_to_pil(transform(original))
    expected = Image.open("tests/assets/ISIC_0000013_2019_transformed.jpg").convert("RGB")
    diff = ImageChops.difference(expected, transformed)
    assert diff.getbbox() is None
