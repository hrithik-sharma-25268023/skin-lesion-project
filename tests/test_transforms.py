"""test cases for train and eval-test transforms"""

from pathlib import Path
from PIL import Image
from skin_lesion_project.processing import image_processing


def test_training_transform_output():
    """Tests the training transforms."""

    image_path = Path(__file__).parent / "assets" / "ISIC_0000013_2019.jpg"
    original = Image.open(image_path).convert("RGB")
    transform = image_processing.training_data_transforms()
    transformed = image_processing.tensor_to_pil(transform(original))
    assert transformed.size == (224, 224)
    assert transformed.mode == "RGB"
