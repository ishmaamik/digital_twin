"""
Consistency, not just average: a box plot of each model's final
(200-sample) top-2 accuracy distribution across all 32 comparisons. The
grand-average script (plot_06) already shows the mean +/- std band across
the sweep; this shows the full distribution shape at the final sample
count -- whether a model is reliably good everywhere, or great on some
comparisons and poor on others.

Outputs (written to result/diagrams/cap500_study/):
    accuracy_distribution_boxplot.png

Run from the repository root:
    python diagrams/cap500_study/plot_13_accuracy_distribution_boxplot.py
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    MODEL_ORDER, MODEL_STYLE, RANDOM_CHANCE_TOP2, INK_PRIMARY, INK_SECONDARY, INK_MUTED, SURFACE,
    load_all_models, all_comparison_labels, ensure_out_dir,
)


def plot(all_results):
    labels = all_comparison_labels(all_results)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color="#e1e0d9", linewidth=1, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.axhline(RANDOM_CHANCE_TOP2, color=INK_MUTED, linestyle=":", linewidth=1.2, zorder=1,
               label="Random chance (top-2 of 16 beams)")

    box_data = [[all_results[m][l]["top2_accuracy"][-1] for l in labels] for m in MODEL_ORDER]
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": INK_PRIMARY, "markersize": 6})
    for patch, model_name in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_STYLE[model_name]["color"])
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color(INK_PRIMARY)

    ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
    ax.set_xticklabels([MODEL_STYLE[m]["label"] for m in MODEL_ORDER], fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylabel("Final (200-sample) top-2 accuracy (%)", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylim(0, 100)
    ax.set_title(f"Consistency across all {len(labels)} comparisons, by model\n"
                 "(box = interquartile range, line = median, diamond = mean)",
                 fontsize=12.5, color=INK_PRIMARY, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "accuracy_distribution_boxplot.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot(load_all_models())
