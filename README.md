# Skin Lesion Project

A modular deep learning package for multi-class skin lesion classification using Convolutional Neural Networks (CNNs) and Transformer-based architectures.

---

## Overview

This repository contains the core implementation of models, training pipelines, and evaluation utilities used for comparing CNN and Transformer architectures on dermoscopic image classification tasks.

The design follows a modular and reusable structure, allowing seamless integration into experimental workflows.

---

## Features

* CNN architectures (ResNet, EfficientNet)
* Transformer-based models (Vision Transformer, Swin Transformer)
* Unified training pipeline
* Transfer learning support (ImageNet pretrained weights)
* Evaluation metrics (Accuracy, F1-score, Balanced Accuracy, Precision, Recall, ROC-AUC)
* Clean and extensible codebase

---

## Project Structure

```
skin_lesion_project/
├── models/            # CNN and Transformer architectures
├── training/          # Training pipeline
├── evaluation/        # Metrics and evaluation
└── utils/             # Helper functions
```

---

## Installation

Clone the repository:

```bash
git clone git@github.com:hrithik-sharma-25268023/skin-lesion-project.git
cd skin-lesion-project
```

Install as a package:

```bash
python setup.py develop
pip install -e .
```

---

## Models Included

* ResNet
* EfficientNet
* Vision Transformer (ViT)
* Swin Transformer

---

## Objective

To provide a fair and reproducible comparison between CNN and Transformer models for multi-class skin lesion classification.

---

## License

This project is for academic and research purposes.
