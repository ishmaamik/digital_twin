"""
Recovery curves and final-accuracy bars for the 10 individually-narrated
Stage 2 + Stage 5 comparisons, using the NEW result data:
  - MLP: result/MLP/ (rehearsal-trained, full source pool)
  - k-NN / Random Forest / Fourier+k-NN: result/new-knn-rf-fknn/
    (rehearsal-trained, source pool capped at 500 samples per comparison)

This is a fresh build against the new data layout -- it does not reuse
diagrams/plot_01_stage2_model_comparison.py or plot_02_stage5_model_comparison.py,
which read the older result/summaries/*.json files.

Outputs (written to result/diagrams/cap500_study/):
    curated_recovery_curves.png     -- 2x5 grid
    curated_final_accuracy_bars.png -- 2x5 grid

Run from the repository root:
    python diagrams/cap500_study/plot_01_curated_recovery_curves.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    MODEL_ORDER, MODEL_STYLE, CURATED_LABELS, CURATED_NICE, RANDOM_CHANCE_TOP2,
    INK_PRIMARY, INK_SECONDARY, INK_MUTED, SURFACE,
    load_all_models, style_axes, ensure_out_dir,
)


def plot_recovery_curves(all_results):
    fig, axes = plt.subplots(2, 5, figsize=(24, 9.5), facecolor=SURFACE)
    fig.suptitle("Top-2 accuracy vs. real target samples -- 10 Stage 2 + Stage 5 comparisons\n"
                 "(MLP: full-source rehearsal; k-NN/RF/Fourier+k-NN: 500-sample source-capped rehearsal)",
                 fontsize=12.5, color=INK_PRIMARY, y=0.99)

    for ax, label in zip(axes.flat, CURATED_LABELS):
        style_axes(ax)
        ax.set_title(CURATED_NICE[label], fontsize=9.5, color=INK_PRIMARY, pad=8)
        ax.axhline(RANDOM_CHANCE_TOP2, color=INK_MUTED, linestyle=":", linewidth=1.2, zorder=1)

        for model_name in MODEL_ORDER:
            data = all_results[model_name][label]
            style = MODEL_STYLE[model_name]
            ax.plot(data["points"], data["top2_accuracy"],
                     color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                     linewidth=style["linewidth"], markersize=5.5, markeredgecolor=SURFACE,
                     markeredgewidth=0.8, zorder=style["zorder"])

        ax.set_xlabel("Real target samples", fontsize=8.5, color=INK_SECONDARY)
        ax.set_ylabel("Top-2 accuracy (%)", fontsize=8.5, color=INK_SECONDARY)
        ax.set_ylim(0, 100)

    handles = [plt.Line2D([0], [0], color=MODEL_STYLE[m]["color"], marker=MODEL_STYLE[m]["marker"],
                           linestyle=MODEL_STYLE[m]["linestyle"], linewidth=MODEL_STYLE[m]["linewidth"],
                           markersize=7, label=MODEL_STYLE[m]["label"])
               for m in MODEL_ORDER]
    handles.append(plt.Line2D([0], [0], color=INK_MUTED, linestyle=":", linewidth=1.2,
                               label="Random chance (top-2 of 16 beams)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.09, 1, 0.93])
    out_path = os.path.join(ensure_out_dir(), "curated_recovery_curves.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_final_bars(all_results):
    fig, axes = plt.subplots(2, 5, figsize=(24, 9), facecolor=SURFACE)
    fig.suptitle("Top-2 accuracy at 200 real target samples -- 10 Stage 2 + Stage 5 comparisons",
                 fontsize=12.5, color=INK_PRIMARY, y=0.98)

    for ax, label in zip(axes.flat, CURATED_LABELS):
        style_axes(ax)
        ax.grid(axis="y", color="#e1e0d9", linewidth=1, zorder=0)
        ax.grid(axis="x", visible=False)
        ax.set_title(CURATED_NICE[label], fontsize=9.5, color=INK_PRIMARY, pad=8)
        ax.axhline(RANDOM_CHANCE_TOP2, color=INK_MUTED, linestyle=":", linewidth=1.2, zorder=1)

        bar_models, bar_values, bar_colors = [], [], []
        for model_name in MODEL_ORDER:
            data = all_results[model_name][label]
            bar_models.append(MODEL_STYLE[model_name]["label"].split(" (")[0])
            bar_values.append(data["top2_accuracy"][-1])
            bar_colors.append(MODEL_STYLE[model_name]["color"])

        order = np.argsort(bar_values)[::-1]
        bar_models = [bar_models[i] for i in order]
        bar_values = [bar_values[i] for i in order]
        bar_colors = [bar_colors[i] for i in order]

        x = np.arange(len(bar_models))
        bars = ax.bar(x, bar_values, color=bar_colors, width=0.6, zorder=2)
        for rect, val in zip(bars, bar_values):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 1.5, f"{val:.1f}%",
                     ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)

        ax.set_xticks(x)
        ax.set_xticklabels(bar_models, rotation=25, ha="right", fontsize=7.5, color=INK_SECONDARY)
        ax.set_ylabel("Top-2 accuracy (%)", fontsize=8.5, color=INK_SECONDARY)
        ax.set_ylim(0, 100)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(ensure_out_dir(), "curated_final_accuracy_bars.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    all_results = load_all_models()
    plot_recovery_curves(all_results)
    plot_final_bars(all_results)
