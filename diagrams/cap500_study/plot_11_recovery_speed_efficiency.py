"""
A different framing from every other diagram in this folder: not "how
accurate is the model at 200 samples" but "how many real samples does it
need to reach a usable accuracy level at all." For each model and each of
the 32 comparisons, linearly interpolates the sweep curve to find the
smallest number of real target samples at which top-2 accuracy first
reaches a 50% threshold; comparisons that never reach 50% within the
200-sample sweep are counted separately rather than silently excluded.

Outputs (written to result/diagrams/cap500_study/):
    recovery_speed_efficiency.png -- box plot of samples-to-50%, by model
    recovery_speed_efficiency.csv -- the underlying per-comparison values

Run from the repository root:
    python diagrams/cap500_study/plot_11_recovery_speed_efficiency.py
"""
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    MODEL_ORDER, MODEL_STYLE, INK_PRIMARY, INK_SECONDARY, SURFACE,
    load_all_models, all_comparison_labels, ensure_out_dir,
)

THRESHOLD = 50.0  # top-2 accuracy percent


def samples_to_threshold(points, accuracies, threshold):
    """Linear interpolation between sweep points; None if never reached."""
    if accuracies[0] >= threshold:
        return points[0]
    for i in range(1, len(points)):
        if accuracies[i] >= threshold:
            x0, x1 = points[i - 1], points[i]
            y0, y1 = accuracies[i - 1], accuracies[i]
            if y1 == y0:
                return x1
            frac = (threshold - y0) / (y1 - y0)
            return x0 + frac * (x1 - x0)
    return None  # never reached within the swept range


def plot(all_results):
    labels = all_comparison_labels(all_results)
    per_model_values = {}
    never_reached_counts = {}
    csv_rows = []

    for model_name in MODEL_ORDER:
        values = []
        never = 0
        for label in labels:
            data = all_results[model_name][label]
            s = samples_to_threshold(data["points"], data["top2_accuracy"], THRESHOLD)
            csv_rows.append((model_name, label, s if s is not None else "never"))
            if s is None:
                never += 1
            else:
                values.append(s)
        per_model_values[model_name] = values
        never_reached_counts[model_name] = never

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color="#e1e0d9", linewidth=1, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)

    box_data = [per_model_values[m] for m in MODEL_ORDER]
    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5)
    for patch, model_name in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_STYLE[model_name]["color"])
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color(INK_PRIMARY)

    xticklabels = []
    for m in MODEL_ORDER:
        n_never = never_reached_counts[m]
        label = MODEL_STYLE[m]["label"]
        if n_never:
            label += f"\n({n_never}/32 never reach {THRESHOLD:.0f}%)"
        xticklabels.append(label)
    ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
    ax.set_xticklabels(xticklabels, fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel(f"Real target samples needed to first reach {THRESHOLD:.0f}% top-2 accuracy", fontsize=10, color=INK_SECONDARY)
    ax.set_title("Recovery speed: how many real samples until the model is actually usable?\n"
                 f"(box = distribution across the comparisons that DO reach {THRESHOLD:.0f}% within 200 samples)",
                 fontsize=12, color=INK_PRIMARY, pad=12)

    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "recovery_speed_efficiency.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)

    csv_path = os.path.join(ensure_out_dir(), "recovery_speed_efficiency.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "comparison", f"samples_to_{THRESHOLD:.0f}pct"])
        writer.writerows(csv_rows)
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    plot(load_all_models())
