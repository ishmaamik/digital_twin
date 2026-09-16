"""
Zero-shot (0 real target samples) top-2 accuracy, grouped bar chart, for all
four models across the 10 curated Stage 2 + Stage 5 comparisons -- built
fresh against result/MLP/ and result/new-knn-rf-fknn/.

Outputs (written to result/diagrams/cap500_study/):
    zero_shot_degradation.png

Run from the repository root:
    python diagrams/cap500_study/plot_02_zero_shot_degradation.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    MODEL_ORDER, MODEL_STYLE, CURATED_LABELS, CURATED_NICE, RANDOM_CHANCE_TOP2,
    INK_PRIMARY, INK_SECONDARY, INK_MUTED, SURFACE,
    load_all_models, ensure_out_dir,
)


def plot(all_results):
    fig, ax = plt.subplots(figsize=(15, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color="#e1e0d9", linewidth=1, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)

    n_models = len(MODEL_ORDER)
    group_width = 0.8
    bar_width = group_width / n_models
    x = np.arange(len(CURATED_LABELS))

    for j, model_name in enumerate(MODEL_ORDER):
        style = MODEL_STYLE[model_name]
        values = [all_results[model_name][label]["top2_accuracy"][0] for label in CURATED_LABELS]
        offset = (j - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, width=bar_width * 0.92, color=style["color"],
                       label=style["label"], zorder=2)
        for rect, val in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 1.0, f"{val:.0f}",
                     ha="center", va="bottom", fontsize=6.5, color=INK_SECONDARY, rotation=90)

    ax.axhline(RANDOM_CHANCE_TOP2, color=INK_MUTED, linestyle=":", linewidth=1.3, zorder=1,
               label="Random chance (top-2 of 16 beams)")

    ax.set_xticks(x)
    ax.set_xticklabels([CURATED_NICE[l] for l in CURATED_LABELS], rotation=25, ha="right",
                        fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylabel("Zero-shot top-2 accuracy (%)", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylim(0, 100)
    ax.set_title("Zero-shot degradation: top-2 accuracy with 0 real target samples, by model",
                 fontsize=13, color=INK_PRIMARY, pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "zero_shot_degradation.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot(load_all_models())
