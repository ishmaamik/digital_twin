"""
Full-matrix diagram: a heatmap of final (N=200 real target samples) top-2
accuracy for all four models across EVERY comparison that has consistent
rehearsal-protocol data for all four models -- 32 source->target comparisons
in total, spanning Stages 2, 3 (straight-segment counterparts), and 5, plus
several directional-reversal and cross-site comparisons that exist in the
result data but are not individually tabulated as their own thesis section.

Why this diagram: the thesis's Stage 2/5 tables and figures deliberately
focus on a curated subset (10 of the 32) chosen for a specific narrative
(time-of-day vs. site-shift, then lane-width mismatch). This heatmap is the
complementary "show everything, nothing cherry-picked" view -- useful in a
defense if asked "is this pattern general, or did you only show us the
comparisons that worked out."

Also produces a companion CSV of the exact matrix values, so any single
number can be checked/quoted precisely.

Outputs (written to result/diagrams/):
    full_matrix_heatmap.png
    full_matrix_final_accuracy.csv

Run from the repository root:
    python diagrams/plot_03_full_matrix_heatmap.py
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    MODEL_ORDER, MODEL_STYLE, INK_PRIMARY, INK_SECONDARY, SURFACE,
    load_all_models, all_comparison_labels, nice_label, ensure_out_dir,
)


def build_matrix(all_results, labels):
    matrix = np.zeros((len(labels), len(MODEL_ORDER)))
    for i, label in enumerate(labels):
        for j, model_name in enumerate(MODEL_ORDER):
            data = all_results[model_name][label]
            matrix[i, j] = data["top2_accuracy"][-1]  # last sweep point = 200 samples
    return matrix


def plot_heatmap(matrix, labels):
    n_rows = len(labels)
    fig, ax = plt.subplots(figsize=(9, 0.32 * n_rows + 2), facecolor=SURFACE)

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_STYLE[m]["label"] for m in MODEL_ORDER],
                        fontsize=10, color=INK_SECONDARY)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([nice_label(l) for l in labels], fontsize=8, color=INK_SECONDARY)

    for i in range(n_rows):
        for j in range(len(MODEL_ORDER)):
            val = matrix[i, j]
            text_color = "#0b0b0b" if 25 < val < 80 else "#ffffff"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.5, color=text_color)

    ax.set_title("Final top-2 accuracy (%) at 200 real target samples -- all 32 comparisons",
                 fontsize=12.5, color=INK_PRIMARY, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=8, colors=INK_SECONDARY)
    cbar.set_label("Top-2 accuracy (%)", fontsize=9, color=INK_SECONDARY)

    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "full_matrix_heatmap.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


def write_csv(matrix, labels):
    out_path = os.path.join(ensure_out_dir(), "full_matrix_final_accuracy.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "description"] + [MODEL_STYLE[m]["label"] for m in MODEL_ORDER])
        for i, label in enumerate(labels):
            writer.writerow([label, nice_label(label)] + [f"{v:.2f}" for v in matrix[i]])
    print(f"Saved {out_path}")


if __name__ == "__main__":
    all_results = load_all_models()
    labels = all_comparison_labels(all_results)
    print(f"{len(labels)} comparisons common to all {len(MODEL_ORDER)} models")
    matrix = build_matrix(all_results, labels)
    plot_heatmap(matrix, labels)
    write_csv(matrix, labels)
