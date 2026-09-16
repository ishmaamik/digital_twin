"""
New experiment set (found already run, not yet visualized anywhere): does
removing College Avenue's road curvature help the REHEARSAL-TRAINED
NON-PARAMETRIC models, the same way Stage 4 showed it helps the reference
MLP? Stage 4 (sec:stage4straight) only ever tested this for the reference
model; this thesis's own "Suggestions for Future Research" explicitly names
the gap: "a curved-road target ... remains untested under rehearsal."

This script closes that gap. Two sibling scripts already produced the data:
  - rehearsal_study/run_raw_bending_rehearsal.py: k-NN, Random Forest, and
    Fourier-k-NN, rehearsal-trained, on the RAW (curved) Scenario 32/33 data
    -- comparisons 1->32, 2->33, 3->32, 4->33.
  - rehearsal_study/run_straight_cap500_rehearsal.py: the same 4 comparisons
    and models, on the CURVATURE-FREE straight-segment Scenario 32straight/
    33straight data instead.
Both scripts additionally cap the source pool at exactly 500 samples (both
zero-shot and every rehearsal sweep point reuse the same fixed 500-sample
draw), so raw and straight are compared at a matched source-data budget --
a cleaner, more controlled version of the same question than simply
comparing the existing (differently-sized) full-source results.

Data: result/stage1_summary_{model}_rehearsal_raw_bending.json and
result/stage1_summary_{model}_rehearsal_straight_cap500.json for
model in {knn, rf, fourier_knn} -- confirmed present for all three.

Outputs (written to result/diagrams/):
    curvature_cap500_recovery_curves.png   -- 4 comparisons x 3 models, raw vs. straight
    curvature_cap500_final_accuracy_bars.png -- final-accuracy raw-vs-straight bars

Run from the repository root:
    python diagrams/plot_11_curvature_matched_source_cap.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import INK_PRIMARY, INK_SECONDARY, INK_MUTED, SURFACE, RANDOM_CHANCE_TOP2, style_axes, ensure_out_dir

SUMMARY_DIR = os.path.join("result", "summaries") if os.path.isdir(os.path.join("result", "summaries")) else "result"
# The raw_bending / straight_cap500 summaries were found directly under
# result/, not result/summaries/ -- checked at write time; fall back cleanly
# either way so this script works regardless of which one holds them.
if not os.path.exists(os.path.join(SUMMARY_DIR, "stage1_summary_knn_rehearsal_raw_bending.json")):
    SUMMARY_DIR = "result"

MODEL_STYLE = {
    "knn":         {"label": "k-NN",             "color": "#eda100"},
    "rf":          {"label": "Random Forest",    "color": "#e87ba4"},
    "fourier_knn": {"label": "Fourier + k-NN",   "color": "#008300"},
}
MODEL_ORDER = ["knn", "rf", "fourier_knn"]

# (raw label, straight label, nice name)
COMPARISONS = [
    ("s1_to_s32", "s1_to_s32straight", "McAllister day -> College Ave day"),
    ("s2_to_s33", "s2_to_s33straight", "McAllister night -> College Ave night"),
    ("s3_to_s32", "s3_to_s32straight", "Rural Road day -> College Ave day"),
    ("s4_to_s33", "s4_to_s33straight", "Rural Road night -> College Ave night"),
]


def load_all():
    data = {}
    for model in MODEL_ORDER:
        raw = json.load(open(os.path.join(SUMMARY_DIR, f"stage1_summary_{model}_rehearsal_raw_bending.json")))
        straight = json.load(open(os.path.join(SUMMARY_DIR, f"stage1_summary_{model}_rehearsal_straight_cap500.json")))
        data[model] = {"raw": raw, "straight": straight}
    return data


def plot_recovery_curves(data):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), facecolor=SURFACE)
    fig.suptitle("Does removing road curvature help the rehearsal-trained non-parametric models?\n"
                 "(500-sample source cap, matched between raw and straight-segment data)",
                 fontsize=12.5, color=INK_PRIMARY, y=0.99)

    for ax, (raw_label, straight_label, nice_name) in zip(axes.flat, COMPARISONS):
        style_axes(ax)
        ax.set_title(nice_name, fontsize=10.5, color=INK_PRIMARY, pad=8)
        ax.axhline(RANDOM_CHANCE_TOP2, color=INK_MUTED, linestyle=":", linewidth=1.2, zorder=1)

        for model in MODEL_ORDER:
            style = MODEL_STYLE[model]
            raw_d = data[model]["raw"]["comparisons"][raw_label]
            straight_d = data[model]["straight"]["comparisons"][straight_label]
            points = data[model]["raw"]["sweep_points"]
            ax.plot(points, [v * 100 for v in raw_d["top2_accuracy"]],
                     color=style["color"], linestyle="--", linewidth=1.8, marker="o", markersize=4,
                     alpha=0.75, label=f"{style['label']} (raw, curved)" if ax is axes.flat[0] else None)
            ax.plot(points, [v * 100 for v in straight_d["top2_accuracy"]],
                     color=style["color"], linestyle="-", linewidth=2.2, marker="^", markersize=4.5,
                     label=f"{style['label']} (straight)" if ax is axes.flat[0] else None)

        ax.set_xlabel("Real target samples", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("Top-2 accuracy (%)", fontsize=9, color=INK_SECONDARY)
        ax.set_ylim(0, 100)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    out_path = os.path.join(ensure_out_dir(), "curvature_cap500_recovery_curves.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_final_bars(data):
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=SURFACE)
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
    bar_width = group_width / (n_models * 2)
    x = np.arange(len(COMPARISONS))

    for j, model in enumerate(MODEL_ORDER):
        style = MODEL_STYLE[model]
        raw_vals, straight_vals = [], []
        for raw_label, straight_label, _ in COMPARISONS:
            raw_vals.append(data[model]["raw"]["comparisons"][raw_label]["top2_accuracy"][-1] * 100)
            straight_vals.append(data[model]["straight"]["comparisons"][straight_label]["top2_accuracy"][-1] * 100)
        offset_raw = (2 * j - n_models + 0.5) * bar_width
        offset_straight = offset_raw + bar_width
        ax.bar(x + offset_raw, raw_vals, width=bar_width * 0.9, color=style["color"], alpha=0.55,
               label=f"{style['label']} (raw)")
        ax.bar(x + offset_straight, straight_vals, width=bar_width * 0.9, color=style["color"],
               label=f"{style['label']} (straight)")

    ax.set_xticks(x)
    ax.set_xticklabels([c[2] for c in COMPARISONS], rotation=15, ha="right", fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylabel("Top-2 accuracy at 200 real target samples (%)", fontsize=10, color=INK_SECONDARY)
    ax.set_ylim(0, 100)
    ax.set_title("Final accuracy, raw (curved) vs. straight-segment data, at a matched 500-sample source cap",
                 fontsize=12, color=INK_PRIMARY, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK_SECONDARY, ncol=2)

    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "curvature_cap500_final_accuracy_bars.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    data = load_all()
    plot_recovery_curves(data)
    plot_final_bars(data)
