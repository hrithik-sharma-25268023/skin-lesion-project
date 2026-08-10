"""
Training loop for skin lesion classification.
"""
from __future__ import annotations

import copy
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm.auto import tqdm


def _rand_bbox(size, lam):
    H, W = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)

    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mixup_cutmix(images, labels, num_classes, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
    """
    Randomly applies either MixUp or CutMix to a batch. Returns
    soft (one-hot mixed) targets so a plain CrossEntropyLoss with
    `label_smoothing` still works fine against them via soft-target CE.
    """
    targets_onehot = F.one_hot(labels, num_classes).float()

    if random.random() > prob:
        return images, targets_onehot

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    if random.random() < 0.5:
        lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
        mixed_images = lam * images + (1 - lam) * images[index]
    else:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha) if cutmix_alpha > 0 else 1.0
        x1, y1, x2, y2 = _rand_bbox(images.size(), lam)
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))

    mixed_targets = lam * targets_onehot + (1 - lam) * targets_onehot[index]
    return mixed_images, mixed_targets


def _soft_ce_loss(logits, soft_targets, label_smoothing=0.0):
    num_classes = logits.size(-1)
    if label_smoothing > 0:
        soft_targets = soft_targets * (1 - label_smoothing) + label_smoothing / num_classes
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def _run_epoch(model, loader, criterion, device, num_classes, optimizer=None,
                scaler=None, use_mixup=False, mixup_alpha=0.2, cutmix_alpha=1.0,
                mixup_prob=0.5, label_smoothing=0.0, grad_clip=1.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False, desc="train" if is_train else "val"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            use_amp = scaler is not None and device.type == "cuda"
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                if is_train and use_mixup:
                    mixed_images, soft_targets = mixup_cutmix(
                        images, labels, num_classes, mixup_alpha, cutmix_alpha, mixup_prob
                    )
                    logits = model(mixed_images)
                    loss = _soft_ce_loss(logits, soft_targets, label_smoothing)
                else:
                    logits = model(images)
                    loss = criterion(logits, labels)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = running_loss / total
    accuracy = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, macro_f1


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs: int,
    checkpoint_path: str,
    early_stopping_patience: int = 7,
    num_classes: int = 8,
    use_amp: bool = True,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mixup_prob: float = 0.5,
    label_smoothing: float = 0.1,
    grad_clip: float = 1.0,
    warmup_epochs: int = 0,
    monitor: str = "macro_f1",
):
    """
    Train `model` and return a history dict with keys:
    train_loss, val_loss, train_accuracy, val_accuracy, macro_f1, lr.

    Parameters
    ----------
    monitor : {"macro_f1", "val_accuracy", "val_loss"}
        Metric used for early stopping and best-checkpoint selection.
        `macro_f1` is recommended for imbalanced multi-class problems
        like this one (rare classes like DF/VASC otherwise get ignored).
    warmup_epochs : int
        Number of epochs to linearly warm the LR up from ~0 to each
        param group's base LR before `scheduler.step()` takes over.
    use_mixup : bool
        Applies MixUp/CutMix during training only (never at eval time).
    """
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "macro_f1": [], "lr": [],
    }

    best_score = -float("inf") if monitor != "val_loss" else float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * warmup_factor

        train_loss, train_acc, _ = _run_epoch(
            model, train_loader, criterion, device, num_classes,
            optimizer=optimizer, scaler=scaler,
            use_mixup=use_mixup, mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha,
            mixup_prob=mixup_prob, label_smoothing=label_smoothing, grad_clip=grad_clip,
        )
        val_loss, val_acc, val_macro_f1 = _run_epoch(
            model, val_loader, criterion, device, num_classes, optimizer=None,
        )

        if epoch >= warmup_epochs and scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["macro_f1"].append(val_macro_f1)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch + 1:>3}/{epochs} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f} macro_f1 {val_macro_f1:.4f} | "
            f"lr {current_lr:.2e}"
        )

        score = {"macro_f1": val_macro_f1, "val_accuracy": val_acc, "val_loss": val_loss}[monitor]
        improved = (score > best_score) if monitor != "val_loss" else (score < best_score)

        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
            patience_counter = 0
            print(f"  -> new best ({monitor}={best_score:.4f}), checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1} "
                      f"(no improvement in {monitor} for {early_stopping_patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
