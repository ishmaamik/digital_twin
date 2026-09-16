"""
Top-2 relative received power vs. real target samples, for the subset of
the 10 curated comparisons that are NOT affected by the Scenario
32straight/33straight NaN artifact (checked directly against both
result/MLP/ and result/new-knn-rf-fknn/ on 2026-09-17 -- see
RELATIVE_POWER_NAN_LABELS in common.py). Of the 10 curated comparisons, 6
are usable: Stage 2's original 4, plus Stage 5's two Site-7 comparisons.

Outputs (written to result/diagrams/cap500_study/):
    relative_power.png -- 2x3 grid

Run from the repository root:
    python diagrams/cap500_study/plot_03_relative_power.py
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    MODEL_ORDER, MODEL_STYLE, STAGE2_LABELS, STAGE2_NICE, STAGE5_NICE,
    RELATIVE_POWER_NAN_LABELS, INK_PRIMARY, INK_SECONDARY, SURFACE,
    load_all_models, style_axes, ensure_out_dir,
)

VALID_STAGE5_LABELS = ["s1_to_s7", "s2_to_s7"]  # the only Stage 5 comparisons without the NaN artifact
ALL_LABELS = STAGE2_LABELS + VALID_STAGE5_LABELS
ALL_NICE = {**STAGE2_NICE, **STAGE5_NICE}


def plot(all_results):
    for label in ALL_LABELS:
        assert label not in RELATIVE_POWER_NAN_LABELS, f"{label} is supposed to be NaN-free"

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), facecolor=SURFACE)
    fig.suptitle("Top-2 relative received power vs. real target samples\n"
                 "(6 of the 10 curated comparisons -- the other 4 are unusable for this metric, see script docstring)",
                 fontsize=12.5, color=INK_PRIMARY, y=0.99)

    for ax, label in zip(axes.flat, ALL_LABELS):
        style_axes(ax)
        ax.set_title(ALL_NICE[label], fontsize=9.5, color=INK_PRIMARY, pad=8)

        for model_name in MODEL_ORDER:
            data = all_results[model_name][label]
            style = MODEL_STYLE[model_name]
            ax.plot(data["points"], data["top2_relative_power"],
                     color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                     linewidth=style["linewidth"], markersize=5.5, markeredgecolor=SURFACE,
                     markeredgewidth=0.8, zorder=style["zorder"])

        ax.set_xlabel("Real target samples", fontsize=8.5, color=INK_SECONDARY)
        ax.set_ylabel("Top-2 relative power (%)", fontsize=8.5, color=INK_SECONDARY)
        ax.set_ylim(0, 105)

    handles = [plt.Line2D([0], [0], color=MODEL_STYLE[m]["color"], marker=MODEL_STYLE[m]["marker"],
                           linestyle=MODEL_STYLE[m]["linestyle"], linewidth=MODEL_STYLE[m]["linewidth"],
                           markersize=7, label=MODEL_STYLE[m]["label"])
               for m in MODEL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.09, 1, 0.93])
    out_path = os.path.join(ensure_out_dir(), "relative_power.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot(load_all_models())
