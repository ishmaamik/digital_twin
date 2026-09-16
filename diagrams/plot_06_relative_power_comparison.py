"""
Relative-power diagram: the same Stage 2 + Stage 5 comparisons (10 total),
but plotting top-2 RELATIVE POWER instead of top-2 accuracy -- i.e. the
fraction of the best-possible received beamforming power that the model's
top-2 predicted beams actually deliver, averaged over the test set. This
metric is already computed and stored in every summary JSON
("top2_relative_power") but was never plotted anywhere in this project
before this script.

Why this diagram: top-2 accuracy alone can understate a model that is
"wrong but close" (predicts a beam adjacent to the true best one, which
still captures most of the received power) or overstate a model that is
"right sometimes, badly wrong otherwise." Relative power is the metric an
O-RAN operator actually cares about -- link quality -- so showing that the
model ranking is consistent under both metrics (or, if it is not,
highlighting exactly where they disagree) is a stronger, more complete
defense of the accuracy-based headline results.

Outputs (written to result/diagrams/):
    relative_power_stage2_stage5.png   -- 2x5 grid (accuracy metric's
                                           analogue), one panel per comparison

Run from the repository root:
    python diagrams/plot_06_relative_power_comparison.py
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    MODEL_ORDER, MODEL_STYLE, STAGE2_LABELS, STAGE2_NICE, STAGE5_LABELS, STAGE5_NICE,
    INK_PRIMARY, INK_SECONDARY, INK_MUTED, SURFACE,
    load_all_models, style_axes, ensure_out_dir,
)

ALL_LABELS = STAGE2_LABELS + STAGE5_LABELS
ALL_NICE = {**STAGE2_NICE, **STAGE5_NICE}


def plot_relative_power(all_results):
    fig, axes = plt.subplots(2, 5, figsize=(24, 9.5), facecolor=SURFACE)
    fig.suptitle("Top-2 relative received power vs. real target samples (Stage 2 + Stage 5)",
                 fontsize=13.5, color=INK_PRIMARY, y=0.98)

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
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_path = os.path.join(ensure_out_dir(), "relative_power_stage2_stage5.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    all_results = load_all_models()
    plot_relative_power(all_results)
