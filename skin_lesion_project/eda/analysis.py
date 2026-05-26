"""module for Exploratory Data Analysis of the Dataset"""

from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib import gridspec as gsc
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from tqdm import tqdm

from skin_lesion_project.utils.eda_utils import LABEL_COLS, PALETTE, CLASS_NAMES


def plot_class_distribution(data: pd.DataFrame):
    """Plots the bar chart and pie chart of the classes."""

    counts = data[LABEL_COLS].sum().rename(CLASS_NAMES).sort_values(ascending=False)
    pct = (counts / len(data) * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Class Distribution", fontsize=15, fontweight="bold")

    # Bar chart
    bars = axes[0].bar(counts.index, counts.values, color=PALETTE, edgecolor="white", linewidth=0.8)
    axes[0].set_title("Absolute count per class")
    axes[0].set_ylabel("Number of images")
    axes[0].tick_params(axis="x", rotation=35)

    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, str(int(val)), ha="center", va="bottom", fontsize=9)

    # Pie chart WITHOUT labels on chart
    wedges, _ = axes[1].pie(counts.values, colors=PALETTE, startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    axes[1].set_title("Proportion per class")

    # Legend with class name + percentage
    legend_labels = [f"{label} ({percent}%)" for label, percent in zip(counts.index, pct.values)]
    axes[1].legend(wedges, legend_labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    return counts, pct
