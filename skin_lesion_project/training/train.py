"""
Training utilities for skin lesion classification models.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.amp import GradScaler, autocast
from tqdm import tqdm


def train(model: nn.Module, train_loader, val_loader,
          criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler=None,
          device: torch.device = torch.device("cpu"), epochs: int = 50,
          checkpoint_path: str = "checkpoints/best_model.pt"):
    """
    Train a classification model.

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    val_loader : DataLoader
    criterion : nn.Module
    optimizer : Optimizer
    scheduler : optional
    device : torch.device
    epochs : int
    checkpoint_path : str

    Returns
    -------
    dict
        Training history.
    """
    model.to(device)
    scaler = GradScaler(enabled=device.type == "cuda")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [], "precision": [], "recall": [], "macro_f1": []}
    best_macro_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            train_predictions.extend(predictions.cpu().numpy())

            train_targets.extend(labels.cpu().numpy())
            progress.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_loader.dataset)
        train_accuracy = accuracy_score(train_targets, train_predictions)

        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type=device.type, enabled=device.type == "cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                predictions = outputs.argmax(dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        precision = precision_score(val_targets, val_predictions, average="macro", zero_division=0)
        recall = recall_score(val_targets, val_predictions, average="macro", zero_division=0)
        macro_f1 = f1_score(val_targets, val_predictions, average="macro", zero_division=0)

        if scheduler is not None:
            scheduler.step()

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), checkpoint_path)
            print("Best model saved.")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["precision"].append(precision)
        history["recall"].append(recall)
        history["macro_f1"].append(macro_f1)

        print(f"Epoch {epoch+1}/{epochs}")

        print(f"Train Loss : {train_loss}")
        print(f"Val Loss   : {val_loss}")

        print(f"Train Acc  : {train_accuracy}")
        print(f"Val Acc    : {val_accuracy}")

        print(f"Precision  : {precision}")
        print(f"Recall     : {recall}")
        print(f"Macro F1   : {macro_f1}")

    print(f"Best Validation Macro F1 : {best_macro_f1}")
    return history
