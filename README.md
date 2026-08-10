# Skin Lesion Project

A modular deep learning package and a Streamlit UI for multi-class skin lesion classification using Convolutional Neural Networks (CNNs) and Transformer-based architectures.

---

## Overview

This repository contains the core implementation of models, training pipelines and Streamlit UI used for comparing CNN and Transformer architectures on dermoscopic image classification tasks.

## Features

* CNN architectures (ResNet, EfficientNet)
* Transformer-based models (Vision Transformer, Swin Transformer)
* Transfer learning support (ImageNet pretrained weights)
---

## Installation

Clone the repository:

```bash
git clone git@github.com:hrithik-sharma-25268023/skin-lesion-project.git
cd skin-lesion-project
```
Create the environment:

```bash
conda env create --file environment.yml
conda activate skin-lesion-project
```

Install as a package:

```bash
python setup.py develop
pip install -e .
```

---

## Objective

To provide a User Interface for comparison between CNN and Transformer models for multi-class skin lesion classification.

---

## License

This project is for academic and research purposes.
