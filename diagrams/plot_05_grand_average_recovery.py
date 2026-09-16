"""
"Grand average" diagram: one figure, one line per model, showing top-2
accuracy vs. real target samples averaged across ALL 32 comparisons (not
just Stage 2's four or Stage 5's six). A shaded band shows +/-1 standard
deviation across comparisons at each sweep point, computed directly from
the per-comparison values already in the JSON summaries -- not a fabricated
uncertainty estimate.

Why this diagram: this is the single "big picture" slide for a defense --
it answers "on average, across every environmental shift we tested, which
model recovers fastest and highest" in one image, and makes the shape of
the finding (the non-parametric models' early advantage, the mlp's slower
but sometimes competitive-at-scale climb) visible without requiring the
audience to scan a 32-row heatmap.

Outputs (written to result/diagrams/):
    grand_average_recovery.png

Run from the repository root:
    python diagrams/plot_05_grand_average_recovery.py
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
    """Returns {model: (points, mean_acc_per_point, std_acc_per_point)}."""
    out = {}
    for model_name in MODEL_ORDER:
        points = all_results[model_name][labels[0]]["points"]
        stacked = np.array([all_results[model_name][label]["top2_accuracy"] for label in labels])
        out[model_name] = (points, stacked.mean(axis=0), stacked.std(axis=0))
    return out


def plot_grand_average(grand_avg, n_comparisons):
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
    ax.set_title(f"Grand-average recovery: top-2 accuracy averaged across all {n_comparisons} comparisons\n"
                 f"(shaded band: +/-1 std. dev. across comparisons)",
                 fontsize=12.5, color=INK_PRIMARY, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "grand_average_recovery.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    all_results = load_all_models()
    labels = all_comparison_labels(all_results)
    grand_avg = compute_grand_average(all_results, labels)
    plot_grand_average(grand_avg, len(labels))
