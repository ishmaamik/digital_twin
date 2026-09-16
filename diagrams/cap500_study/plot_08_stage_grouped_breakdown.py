"""
Groups all 32 comparisons into four thesis-aligned categories and shows
mean final (200-sample) accuracy per model, per category, as a grouped bar
chart -- a single "which situation is hardest, for which model" overview.

Categories (mutually exclusive, cover all 32):
  A. Stage 2 core (4)       -- s1_to_s2, s3_to_s4, s1_to_s3, s2_to_s4
  B. Stage 5 extension (6)  -- McAllister (day/night) -> Site 7 / College Ave
  C. Reverse-direction (4)  -- s2_to_s1, s4_to_s2, s7_to_s1, s7_to_s2
                               (the reversed counterparts of 4 of the above)
  D. Extended matrix (18)   -- every other comparison in the 32 (mostly
                               Rural Road / College Ave / Site 7 combinations
                               never individually narrated in the thesis)

Outputs (written to result/diagrams/cap500_study/):
    stage_grouped_breakdown.png

Run from the repository root:
    python diagrams/cap500_study/plot_08_stage_grouped_breakdown.py
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

GROUP_A = ["s1_to_s2", "s3_to_s4", "s1_to_s3", "s2_to_s4"]
GROUP_B = ["s1_to_s7", "s1_to_s32straight", "s1_to_s33straight", "s2_to_s7", "s2_to_s32straight", "s2_to_s33straight"]
GROUP_C = ["s2_to_s1", "s4_to_s2", "s7_to_s1", "s7_to_s2"]
GROUPS = {"A: Stage 2 core\n(4 comparisons)": GROUP_A,
          "B: Stage 5 extension\n(6 comparisons)": GROUP_B,
          "C: Reverse-direction\n(4 comparisons)": GROUP_C}


def plot(all_results):
    labels = all_comparison_labels(all_results)
    already_grouped = set(GROUP_A) | set(GROUP_B) | set(GROUP_C)
    group_d = [l for l in labels if l not in already_grouped]
    GROUPS["D: Extended matrix\n(%d comparisons)" % len(group_d)] = group_d
    assert sum(len(v) for v in GROUPS.values()) == 32

    fig, ax = plt.subplots(figsize=(12, 7), facecolor=SURFACE)
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

    group_names = list(GROUPS.keys())
    n_models = len(MODEL_ORDER)
    bar_width = 0.8 / n_models
    x = np.arange(len(group_names))

    for j, model_name in enumerate(MODEL_ORDER):
        style = MODEL_STYLE[model_name]
        means = []
        for g in group_names:
            vals = [all_results[model_name][l]["top2_accuracy"][-1] for l in GROUPS[g]]
            means.append(np.mean(vals))
        offset = (j - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, width=bar_width * 0.9, color=style["color"], label=style["label"], zorder=2)
        for rect, val in zip(bars, means):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 1.2, f"{val:.1f}", ha="center", va="bottom",
                     fontsize=7.5, color=INK_SECONDARY, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("Mean final top-2 accuracy (%)", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylim(0, 100)
    ax.set_title("Final accuracy by comparison category, by model\n"
                 "(mean across the comparisons in each category, at 200 real target samples)",
                 fontsize=12.5, color=INK_PRIMARY, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, ncol=2)

    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "stage_grouped_breakdown.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot(load_all_models())
