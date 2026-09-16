"""
Zero-shot degradation diagram: for the ten Stage 2 + Stage 5 comparisons,
a grouped bar chart of each model's ZERO-SHOT (0 real target samples)
top-2 accuracy -- i.e. how badly each model degrades the instant it is
pointed at a new environment, before any rehearsal data is available at
all -- with the 12.5% random-chance line overlaid.

Why this diagram: the recovery curves show the whole sweep, but a defense
audience often wants the single starkest number first: "how bad is it on
day one, with zero real data." This isolates exactly that point for every
model and every comparison in one chart, motivating why the rest of the
thesis (rehearsal, model choice) matters at all -- if zero-shot already
worked, none of the recovery analysis would be necessary.

Outputs (written to result/diagrams/):
    zero_shot_degradation.png

Run from the repository root:
    python diagrams/plot_07_zero_shot_degradation.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    MODEL_ORDER, MODEL_STYLE, STAGE2_LABELS, STAGE2_NICE, STAGE5_LABELS, STAGE5_NICE,
    RANDOM_CHANCE_TOP2, INK_PRIMARY, INK_SECONDARY, INK_MUTED, SURFACE,
    load_all_models, ensure_out_dir,
)

ALL_LABELS = STAGE2_LABELS + STAGE5_LABELS
ALL_NICE = {**STAGE2_NICE, **STAGE5_NICE}


def plot_zero_shot(all_results):
    fig, ax = plt.subplots(figsize=(15, 7), facecolor=SURFACE)
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
    bar_width = group_width / n_models
    x = np.arange(len(ALL_LABELS))

    for j, model_name in enumerate(MODEL_ORDER):
        style = MODEL_STYLE[model_name]
        values = [all_results[model_name][label]["top2_accuracy"][0] for label in ALL_LABELS]  # index 0 = 0 samples
        offset = (j - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, width=bar_width * 0.92, color=style["color"],
                       label=style["label"], zorder=2)
        for rect, val in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 1.0, f"{val:.0f}",
                     ha="center", va="bottom", fontsize=6.5, color=INK_SECONDARY, rotation=90)

    ax.axhline(RANDOM_CHANCE_TOP2, color=INK_MUTED, linestyle=":", linewidth=1.3, zorder=1,
               label="Random chance (top-2 of 16 beams)")

    ax.set_xticks(x)
    ax.set_xticklabels([ALL_NICE[l] for l in ALL_LABELS], rotation=25, ha="right", fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylabel("Zero-shot top-2 accuracy (%)", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylim(0, 100)
    ax.set_title("Zero-shot degradation: top-2 accuracy with 0 real target samples, by model",
                 fontsize=13, color=INK_PRIMARY, pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncol=1)

    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "zero_shot_degradation.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    all_results = load_all_models()
    plot_zero_shot(all_results)
