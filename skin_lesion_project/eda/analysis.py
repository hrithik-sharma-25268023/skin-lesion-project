"""module for Exploratory Data Analysis of the Dataset"""

import os
import random
from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from skin_lesion_project.utils.eda_utils import LABEL_COLS, PALETTE, CLASS_NAMES


def plot_class_distribution(data: pd.DataFrame) -> Tuple:
    """Plots class counts and proportions."""

    counts = data[LABEL_COLS].sum().rename(CLASS_NAMES).sort_values(ascending=False)
    pct = (counts / counts.sum() * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Class Distribution", fontsize=15, fontweight="bold")

    bars = axes[0].bar(counts.index, counts.values,
                       color=PALETTE, edgecolor="white", linewidth=0.8)

    axes[0].set_title("Absolute count per class")
    axes[0].set_ylabel("Number of Images")
    axes[0].tick_params(axis="x", rotation=35)

    offset = counts.max() * 0.01

    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{int(val)}",
            ha="center", va="bottom", fontsize=9)

    wedges, _ = axes[1].pie(counts.values,
        colors=PALETTE, startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    axes[1].set_title("Proportion per class")
    legend_labels = [f"{cls} ({p}%)" for cls, p in zip(counts.index, pct.values)]
    axes[1].legend(wedges, legend_labels,
                   title="Classes", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    return counts, pct


def plot_sample_grid(data: pd.DataFrame, image_dir: str, n_per_class: int = 5) -> None:
    """Displays random sample images for each class."""

    n_rows = len(LABEL_COLS)
    fig, axes = plt.subplots(n_rows, n_per_class, figsize=(n_per_class * 2.5, n_rows * 2.8))

    fig.suptitle("Sample Images per Class", fontsize=14, fontweight="bold", y=1.02)

    if n_rows == 1:
        axes = [axes]

    for row_idx, label_col in enumerate(LABEL_COLS):
        class_samples = data[data[label_col] == 1]["image"].tolist()
        if len(class_samples) > n_per_class:
            class_samples = random.sample(class_samples, n_per_class)
        for col_idx in range(n_per_class):
            ax = axes[row_idx][col_idx]
            if col_idx >= len(class_samples):
                ax.axis("off")
                continue
            image_name = class_samples[col_idx]
            jpg_path = os.path.join(image_dir, f"{image_name}.jpg")
            png_path = os.path.join(image_dir, f"{image_name}.png")
            img_path = jpg_path if os.path.exists(jpg_path) else png_path
            try:
                img = (
                    Image.open(img_path)
                    .convert("RGB")
                    .resize((128, 128)))
                ax.imshow(img)
            except Exception:
                ax.set_facecolor("#dddddd")
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(image_name, fontsize=7, labelpad=5)

        axes[row_idx][0].annotate(CLASS_NAMES[label_col], xy=(-0.45, 0.5), xycoords="axes fraction", fontsize=10,
                                  fontweight="bold", ha="right", va="center")

    plt.tight_layout()
    plt.show()


def plot_image_sizes(data: pd.DataFrame, image_dir: str, max_images: int=3000) -> None:
    """Plots distributions of image widths, heights, and aspect ratios from a random sample of dataset images."""

    sample_df = data.sample(min(max_images, len(data)), random_state=42)
    widths, heights, aspects = [], [], []

    for fname in tqdm(sample_df["image"], leave=False):
        try:
            with Image.open(os.path.join(image_dir, fname)) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspects.append(w / h)
        except Exception:
            pass

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Image Size Distribution", fontsize=14, fontweight="bold")

    axes[0].hist(widths,  bins=40, color="#378ADD", edgecolor="white")
    axes[0].set_title("Width distribution")
    axes[0].set_xlabel("pixels")
    axes[0].axvline(np.median(widths), color="red", linestyle="--", label=f"median={int(np.median(widths))}")
    axes[0].legend()
    axes[1].hist(heights, bins=40, color="#1D9E75", edgecolor="white")
    axes[1].set_title("Height distribution")
    axes[1].set_xlabel("pixels")
    axes[1].axvline(np.median(heights), color="red", linestyle="--", label=f"median={int(np.median(heights))}")
    axes[1].legend()
    axes[2].scatter(widths, heights, alpha=0.3, s=8, color="#534AB7")
    axes[2].set_title("Width vs Height")
    axes[2].set_xlabel("Width")
    axes[2].set_ylabel("Height")
    plt.tight_layout()
    print(f"Width median={int(np.median(widths))}, min={min(widths)}, max={max(widths)}")
    print(f"Height median={int(np.median(heights))}, min={min(heights)}, max={max(heights)}")


def plot_pixel_stats(data: pd.DataFrame, image_dir: str, n_samples: int = 500) -> None:
    """Plots the pixel intensity of RGB images"""

    per_class = max(1, n_samples // len(LABEL_COLS))
    sampled_indices = set()
    for col in LABEL_COLS:
        idx = data[data[col] == 1].sample(min(per_class, (data[col]==1).sum()),
                                          random_state=42).index.tolist()
        sampled_indices.update(idx)
    sample_df = data.loc[list(sampled_indices)]
    channel_means = {col: {"R": [], "G": [], "B": []} for col in LABEL_COLS}
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), leave=False):
        dominant_class = LABEL_COLS[np.argmax(row[LABEL_COLS].values)]
        try:
            img = np.array(
                Image.open(os.path.join(image_dir, row["image"])).convert("RGB").resize((128,128)))
            channel_means[dominant_class]["R"].append(img[:,:,0].mean())
            channel_means[dominant_class]["G"].append(img[:,:,1].mean())
            channel_means[dominant_class]["B"].append(img[:,:,2].mean())
        except Exception:
            pass

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Mean Channel Intensity per Class", fontsize=14, fontweight="bold")

    for ch_idx, (ch, color) in enumerate([("R","#E24B4A"),("G","#1D9E75"),("B","#378ADD")]):
        means = [np.mean(channel_means[c][ch]) if channel_means[c][ch] else 0
                 for c in LABEL_COLS]
        stds  = [np.std(channel_means[c][ch])  if channel_means[c][ch] else 0
                 for c in LABEL_COLS]
        short = [CLASS_NAMES[c].split()[0] for c in LABEL_COLS]
        axes[ch_idx].bar(short, means, yerr=stds, color=color,
                         alpha=0.8, edgecolor="white", capsize=4)
        axes[ch_idx].set_title(f"{ch} channel mean ± std")
        axes[ch_idx].set_ylim(0, 255)
        axes[ch_idx].tick_params(axis="x", rotation=35)
        axes[ch_idx].set_ylabel("Pixel intensity (0–255)")

    plt.tight_layout()
