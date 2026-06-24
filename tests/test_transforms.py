"""test cases for train and eval-test transforms"""

from PIL import Image
from skin_lesion_project.processing import image_processing



def test_training_transform_output():
    """tests the train transforms"""

    original = Image.open("/mnt/d/skin-lesion-data/2019+2020/train/ISIC_0000013_2019.jpg").convert("RGB")
    transform = image_processing.training_data_transforms()
    transformed = image_processing.tensor_to_pil(transform(original))
    assert transformed.size == (224, 224)
    assert transformed.mode == "RGB"
