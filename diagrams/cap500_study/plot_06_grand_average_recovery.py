"""
Grand-average recovery: top-2 accuracy vs. real target samples, averaged
across all 32 comparisons, one line per model, with a +/-1 std-dev band
computed directly from the per-comparison values.

Outputs (written to result/diagrams/cap500_study/):
    grand_average_recovery.png

Run from the repository root:
    python diagrams/cap500_study/plot_06_grand_average_recovery.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    MODEL_ORDER, MODEL_STYLE, RANDOM_CHANCE_TOP2,
    INK_PRIMARY, INK_SECONDARY, INK_MUTED, SURFACE,
    load_all_models, all_comparison_labels, style_axes, ensure_out_dir,
)


def compute_grand_average(all_results, labels):
    out = {}
    for model_name in MODEL_ORDER:
        points = all_results[model_name][labels[0]]["points"]
        stacked = np.array([all_results[model_name][label]["top2_accuracy"] for label in labels])
        out[model_name] = (points, stacked.mean(axis=0), stacked.std(axis=0))
    return out


def plot(grand_avg, n):
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=SURFACE)
    style_axes(ax)
    ax.axhline(RANDOM_CHANCE_TOP2, color=INK_MUTED, linestyle=":", linewidth=1.2, zorder=1,
               label="Random chance (top-2 of 16 beams)")

    for model_name in MODEL_ORDER:
        points, mean_acc, std_acc = grand_avg[model_name]
        style = MODEL_STYLE[model_name]
        ax.plot(points, mean_acc, color=style["color"], marker=style["marker"],
                 linestyle=style["linestyle"], linewidth=style["linewidth"] + 0.4,
                 markersize=7, markeredgecolor=SURFACE, markeredgewidth=1,
                 zorder=style["zorder"], label=style["label"])
        ax.fill_between(points, mean_acc - std_acc, mean_acc + std_acc,
                         color=style["color"], alpha=0.10, zorder=style["zorder"] - 1)

    ax.set_xlabel("Real target-domain samples", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylabel("Top-2 accuracy (%), mean across comparisons", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylim(0, 100)
    ax.set_title(f"Grand-average recovery: top-2 accuracy averaged across all {n} comparisons\n"
                 f"(shaded band: +/-1 std. dev. across comparisons)",
                 fontsize=12.5, color=INK_PRIMARY, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "grand_average_recovery.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    all_results = load_all_models()
    labels = all_comparison_labels(all_results)
    plot(compute_grand_average(all_results, labels), len(labels))
