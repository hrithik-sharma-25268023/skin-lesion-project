"""module for Exploratory Data Analysis of the Dataset"""

import os
from pathlib import Path
import random
from typing import Tuple

from matplotlib import pyplot as plt
from matplotlib import gridspec as gsc
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from tqdm import tqdm

from skin_lesion_project.utils.eda_utils import LABEL_COLS, PALETTE, CLASS_NAMES


def plot_class_distribution(data: pd.DataFrame) -> Tuple:
    """Plots the bar chart and pie chart of the classes."""

    counts = data[LABEL_COLS].sum().rename(CLASS_NAMES).sort_values(ascending=False)
    pct = (counts / len(data) * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Class Distribution", fontsize=15, fontweight="bold")

    # Bar Chart
    bars = axes[0].bar(counts.index, counts.values, color=PALETTE, edgecolor="white", linewidth=0.8)
    axes[0].set_title("Absolute count per class")
    axes[0].set_ylabel("Number of images")
    axes[0].tick_params(axis="x", rotation=35)

    # Pie Chart
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, str(int(val)), ha="center", va="bottom", fontsize=9)
    wedges, _ = axes[1].pie(counts.values, colors=PALETTE, startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    axes[1].set_title("Proportion per class")
    legend_labels = [f"{label} ({percent}%)" for label, percent in zip(counts.index, pct.values)]
    axes[1].legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    return counts, pct


def plot_sample_grid(data: pd.DataFrame, image_dir: str, n_per_class: int = 5) -> None:
    """Displays sample images for each class with filenames."""

    n_rows = len(LABEL_COLS)
    fig, axes = plt.subplots(n_rows, n_per_class, figsize=(n_per_class * 2.5, n_rows * 2.8))
    fig.suptitle("Sample Images per Class", fontsize=14, fontweight="bold", y=1.02)
    if n_rows == 1:
        axes = [axes]

    for row_idx, col in enumerate(LABEL_COLS):

        samples = data[data[col] == 1]["image"].tolist()
        samples = random.sample(samples, min(n_per_class, len(samples)))

        for col_idx in range(n_per_class):

            ax = axes[row_idx][col_idx]
            if col_idx >= len(samples):
                ax.axis("off")
                continue

            fname = samples[col_idx]
            img_path = os.path.join(image_dir, fname)

            try:
                img = Image.open(img_path).convert("RGB").resize((128, 128))
                ax.imshow(img)

            except Exception:
                ax.set_facecolor("#ddd")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#999")

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(fname, fontsize=7, labelpad=5)
        axes[row_idx][0].annotate(CLASS_NAMES[col], xy=(-0.45, 0.5), xycoords="axes fraction", fontsize=10, fontweight="bold", ha="right", va="center")

    plt.tight_layout()
    plt.show()
