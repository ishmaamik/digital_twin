"""
BONUS / methodology-robustness check: does capping the rehearsal source
pool at 500 samples (result/new-knn-rf-fknn/, used everywhere else in this
folder) actually change the answer, compared to the OLDER full,
scenario-size-dependent source pool used earlier in this project
(result/summaries/*.json)? This is the one script in this folder that
intentionally looks back at the older data -- not to re-compare models
against each other (already covered elsewhere), but to sanity-check
whether the 500-sample-cap methodology choice itself is safe, i.e.
comparisons made under it are not an artifact of that specific cap.

For every comparison present in both datasets, plots old (full-source)
final accuracy against new (500-cap) final accuracy per model; points near
the y=x diagonal mean the cap barely changes the result.

Outputs (written to result/diagrams/cap500_study/):
    source_budget_sensitivity.png
    source_budget_sensitivity.csv

Run from the repository root:
    python diagrams/cap500_study/plot_12_source_budget_sensitivity.py
"""
import csv
import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import MODEL_STYLE, INK_PRIMARY, INK_SECONDARY, SURFACE, style_axes, ensure_out_dir, load_cap500_model

OLD_SUMMARY_DIR = os.path.join("result", "summaries")
OLD_MODEL_FILES = {
    "knn": [
        "stage1_summary_knn_rehearsal.json",
        "stage1_summary_knn_rehearsal_newsites.json",
        "stage1_summary_knn_rehearsal_stage2.json",
        "stage1_summary_knn_rehearsal_stage3.json",
        "stage1_summary_knn_rehearsal_stage3b.json",
        "stage1_summary_knn_rehearsal_s32straight_to_s7.json",
    ],
    "rf": [
        "stage1_summary_rf_rehearsal.json",
        "stage1_summary_rf_rehearsal_newsites.json",
        "stage1_summary_rf_rehearsal_stage2.json",
        "stage1_summary_rf_rehearsal_stage3.json",
        "stage1_summary_rf_rehearsal_stage3b.json",
        "stage1_summary_rf_rehearsal_s32straight_to_s7.json",
    ],
    "fourier_knn": [
        "stage1_summary_fourier_knn_rehearsal.json",
        "stage1_summary_fourier_knn_rehearsal_newsites.json",
        "stage1_summary_fourier_knn_rehearsal_stage2.json",
        "stage1_summary_fourier_knn_rehearsal_stage3.json",
        "stage1_summary_fourier_knn_rehearsal_stage3b.json",
        "stage1_summary_fourier_knn_rehearsal_s32straight_to_s7.json",
    ],
}
NONPARAMETRIC_MODELS = ["knn", "rf", "fourier_knn"]


def load_old_model(model_name):
    merged = {}
    for fname in OLD_MODEL_FILES[model_name]:
        with open(os.path.join(OLD_SUMMARY_DIR, fname)) as f:
            d = json.load(f)
        for label, vals in d["comparisons"].items():
            if label not in merged:
                merged[label] = vals["top2_accuracy"][-1] * 100  # final (200-sample) value only
    return merged


def plot():
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5), facecolor=SURFACE)

    csv_rows = []
    for ax, model_name in zip(axes, NONPARAMETRIC_MODELS):
        style_axes(ax)
        style = MODEL_STYLE[model_name]
        old_final = load_old_model(model_name)
        new_data = load_cap500_model(model_name)

        common_labels = sorted(set(old_final.keys()) & set(new_data.keys()))
        old_vals = [old_final[l] for l in common_labels]
        new_vals = [new_data[l]["top2_accuracy"][-1] for l in common_labels]
        for l, ov, nv in zip(common_labels, old_vals, new_vals):
            csv_rows.append((model_name, l, ov, nv, abs(ov - nv)))

        ax.plot([0, 100], [0, 100], color="#c3c2b7", linestyle="--", linewidth=1, zorder=1)
        ax.scatter(old_vals, new_vals, color=style["color"], s=40, alpha=0.85, zorder=2,
                   edgecolor=SURFACE, linewidth=0.6)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Final accuracy, full source pool (%)", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("Final accuracy, 500-sample-capped source (%)", fontsize=9, color=INK_SECONDARY)
        ax.set_title(f"{style['label'].split(' (')[0]}\n(n={len(common_labels)} shared comparisons)",
                     fontsize=10.5, color=INK_PRIMARY, pad=8)

    fig.suptitle("Does capping the rehearsal source pool at 500 samples change the answer?",
                 fontsize=13, color=INK_PRIMARY, y=1.03)
    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "source_budget_sensitivity.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)

    csv_path = os.path.join(ensure_out_dir(), "source_budget_sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "comparison", "full_source_final_acc", "capped500_final_acc", "abs_diff"])
        for row in csv_rows:
            writer.writerow([row[0], row[1], f"{row[2]:.2f}", f"{row[3]:.2f}", f"{row[4]:.2f}"])
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    plot()
