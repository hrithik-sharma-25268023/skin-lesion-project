from setuptools import setup, find_packages

setup(
    name="skin-lesion-project",
    version="0.1.0",
    description="Skin lesion classification project using PyTorch",
    author="Hrithik Sharma",
    packages=find_packages(),
    python_requires=">=3.10",

    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scikit-learn",
        "Pillow",
        "opencv-python",
        "matplotlib",
        "plotly",
        "seaborn",
        "tqdm",
        "PyYAML",
        "ipykernel",
        "timm",
        "boto3",
        "haversine",
        "dvc",
        "albumentations",
        "transformers",
    ],

    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ]
    },

    include_package_data=True,
    zip_safe=False,
)
