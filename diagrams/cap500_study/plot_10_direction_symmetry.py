"""
Direction symmetry: every one of the 32 comparisons pairs up with its
reverse (source and target swapped) -- verified directly: all 32 labels
resolve into exactly 16 reversible pairs, 100% coverage. For each model,
plots forward final accuracy against reverse final accuracy as a scatter;
points near the y=x diagonal mean source/target order barely matters for
that pair, points far from it mean direction matters a lot.

This directly extends the thesis's R1/R2 finding ("reversing source and
target confirms the original Stage 1 result") from 2 hand-picked pairs to
all 16 available pairs, for all four models at once.

Outputs (written to result/diagrams/cap500_study/):
    direction_symmetry.png -- one scatter panel per model
    direction_symmetry.csv -- the underlying forward/reverse pairs

Run from the repository root:
    python diagrams/cap500_study/plot_10_direction_symmetry.py
"""
import csv
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    MODEL_ORDER, MODEL_STYLE, INK_PRIMARY, INK_SECONDARY, SURFACE,
    load_all_models, all_comparison_labels, style_axes, ensure_out_dir,
)


def find_reversible_pairs(labels):
    label_set = set(labels)
    seen = set()
    pairs = []
    for label in labels:
        src, tgt = label.split("_to_")
        rev = f"{tgt}_to_{src}"
        if rev in label_set and label not in seen and rev not in seen:
            pairs.append((label, rev))
            seen.add(label)
            seen.add(rev)
    return pairs


def plot(all_results):
    labels = all_comparison_labels(all_results)
    pairs = find_reversible_pairs(labels)
    print(f"{len(pairs)} reversible pairs found, covering {2 * len(pairs)} of {len(labels)} comparisons")

    fig, axes = plt.subplots(2, 2, figsize=(11, 11), facecolor=SURFACE)
    rows_for_csv = []

    for ax, model_name in zip(axes.flat, MODEL_ORDER):
        style_axes(ax)
        style = MODEL_STYLE[model_name]
        forward_vals, reverse_vals = [], []
        for a, b in pairs:
            fv = all_results[model_name][a]["top2_accuracy"][-1]
            rv = all_results[model_name][b]["top2_accuracy"][-1]
            forward_vals.append(fv)
            reverse_vals.append(rv)
            rows_for_csv.append((model_name, a, b, fv, rv, abs(fv - rv)))

        ax.plot([0, 100], [0, 100], color="#c3c2b7", linestyle="--", linewidth=1, zorder=1)
        ax.scatter(forward_vals, reverse_vals, color=style["color"], s=45, zorder=2, alpha=0.85,
                   edgecolor=SURFACE, linewidth=0.6)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Accuracy, arbitrary \"forward\" direction (%)", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("Accuracy, reverse direction (%)", fontsize=9, color=INK_SECONDARY)
        ax.set_title(style["label"], fontsize=11, color=INK_PRIMARY, pad=8)

    fig.suptitle(f"Direction symmetry: forward vs. reverse final accuracy, all {len(pairs)} reversible pairs\n"
                 "(points on the diagonal = source/target order does not matter)",
                 fontsize=12.5, color=INK_PRIMARY, y=1.0)
    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "direction_symmetry.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)

    csv_path = os.path.join(ensure_out_dir(), "direction_symmetry.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "label_a", "label_b", "acc_a", "acc_b", "abs_diff"])
        for row in rows_for_csv:
            writer.writerow([row[0], row[1], row[2], f"{row[3]:.2f}", f"{row[4]:.2f}", f"{row[5]:.2f}"])
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    plot(load_all_models())
