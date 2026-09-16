"""
Full-matrix heatmap: final (200-real-sample) top-2 accuracy for all four
models across all 32 comparisons in the new dataset (result/MLP/ +
result/new-knn-rf-fknn/), plus a companion CSV of the exact values.

Outputs (written to result/diagrams/cap500_study/):
    full_matrix_heatmap.png
    full_matrix_final_accuracy.csv

Run from the repository root:
    python diagrams/cap500_study/plot_04_full_matrix_heatmap.py
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
            matrix[i, j] = all_results[model_name][label]["top2_accuracy"][-1]
    return matrix


def plot_heatmap(matrix, labels):
    n_rows = len(labels)
    fig, ax = plt.subplots(figsize=(9.5, 0.32 * n_rows + 2), facecolor=SURFACE)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_STYLE[m]["label"] for m in MODEL_ORDER],
                        fontsize=9, color=INK_SECONDARY, rotation=12, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([nice_label(l) for l in labels], fontsize=8, color=INK_SECONDARY)

    for i in range(n_rows):
        for j in range(len(MODEL_ORDER)):
            val = matrix[i, j]
            text_color = "#0b0b0b" if 25 < val < 80 else "#ffffff"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.5, color=text_color)

    ax.set_title("Final top-2 accuracy (%) at 200 real target samples -- all 32 comparisons\n"
                 "(MLP: full-source rehearsal; others: 500-sample source-capped rehearsal)",
                 fontsize=12, color=INK_PRIMARY, pad=12)
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
