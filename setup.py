from setuptools import setup, find_packages

setup(
    name="skin-lesion-project",
    version="0.1.0",
    description="Skin lesion classification project using PyTorch",
    author="Hrithik Sharma",
    packages=find_packages(),
    python_requires=">=3.10",

    install_requires=[
        "torch==2.4.1",
        "torchvision==0.19.1",
        "numpy<2",
        "pandas",
        "scikit-learn",
        "Pillow",
        "opencv-python",
        "matplotlib",
        "plotly",
        "seaborn",
        "tqdm",
        "PyYAML",
        "timm",
        "boto3",
        "albumentations",
        "transformers",
    ],

    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "tomli",
        ]
    },

    include_package_data=True,
    zip_safe=False,
)
