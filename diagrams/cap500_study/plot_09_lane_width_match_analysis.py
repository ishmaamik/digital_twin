"""
Extends the thesis's central Stage 3 finding -- "a lane-width mismatch
roughly doubles site-shift severity" (originally shown on 8 hand-picked
comparisons) -- across the FULL 32-comparison matrix, by model.

Every comparison is classified by whether its source and target scenario
share the same lane width:
    NARROW = {s1, s2, s32straight, s33straight}  (McAllister, College Ave)
    WIDE   = {s3, s4, s7}                        (Rural Road, Site 7)
so every one of the 32 comparisons falls into exactly one of "same width"
or "different width".

Outputs (written to result/diagrams/cap500_study/):
    lane_width_match_analysis.png

Run from the repository root:
    python diagrams/cap500_study/plot_09_lane_width_match_analysis.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    MODEL_ORDER, MODEL_STYLE, RANDOM_CHANCE_TOP2, INK_PRIMARY, INK_SECONDARY, INK_MUTED, SURFACE,
    load_all_models, all_comparison_labels, ensure_out_dir,
)

NARROW = {"s1", "s2", "s32straight", "s33straight"}
WIDE = {"s3", "s4", "s7"}


def width_of(scenario):
    if scenario in NARROW:
        return "narrow"
    if scenario in WIDE:
        return "wide"
    raise ValueError(scenario)


def classify(label):
    src, tgt = label.split("_to_")
    return "Same lane width" if width_of(src) == width_of(tgt) else "Different lane width"


def plot(all_results):
    labels = all_comparison_labels(all_results)
    same = [l for l in labels if classify(l) == "Same lane width"]
    diff = [l for l in labels if classify(l) == "Different lane width"]
    print(f"Same-width comparisons: {len(same)}, different-width comparisons: {len(diff)}")

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
    ax.axhline(RANDOM_CHANCE_TOP2, color=INK_MUTED, linestyle=":", linewidth=1.2, zorder=1)

    categories = [f"Same lane width\n(n={len(same)})", f"Different lane width\n(n={len(diff)})"]
    n_models = len(MODEL_ORDER)
    bar_width = 0.8 / n_models
    x = np.arange(len(categories))

    for j, model_name in enumerate(MODEL_ORDER):
        style = MODEL_STYLE[model_name]
        same_vals = [all_results[model_name][l]["top2_accuracy"][-1] for l in same]
        diff_vals = [all_results[model_name][l]["top2_accuracy"][-1] for l in diff]
        means = [np.mean(same_vals), np.mean(diff_vals)]
        offset = (j - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, width=bar_width * 0.9, color=style["color"], label=style["label"], zorder=2)
        for rect, val in zip(bars, means):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 1.2, f"{val:.1f}", ha="center", va="bottom",
                     fontsize=8.5, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("Mean final top-2 accuracy (%)", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylim(0, 100)
    ax.set_title("Does a lane-width mismatch still hurt, across the full matrix?\n"
                 "Mean final accuracy, same- vs. different-lane-width comparisons, by model",
                 fontsize=12.5, color=INK_PRIMARY, pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "lane_width_match_analysis.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot(load_all_models())
