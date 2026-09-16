"""
"Who wins" diagram: across all 32 comparisons with data for all four models,
count how many times each model achieves the single highest final (N=200)
top-2 accuracy, and separately report each model's mean rank (1=best,
4=worst) averaged over all 32 comparisons. Ties (accuracy within 1e-9,
essentially exact ties only) are split equally between the tied models for
the win count.

Why this diagram: the recovery-curve and bar-chart figures show accuracy
comparison-by-comparison; this is the single "which model comes out on top
most often, overall" summary a committee member is likely to ask for
directly. It is computed, not asserted -- every number here is read from
the same JSON summaries as every other script in this folder.

Outputs (written to result/diagrams/):
    model_win_rate.png    -- bar chart: win count out of 32 comparisons, and mean rank
    model_win_rate.csv    -- the underlying per-comparison ranks, for auditing

Run from the repository root:
    python diagrams/plot_04_model_win_rate.py
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


def compute_wins_and_ranks(all_results, labels):
    win_counts = {m: 0.0 for m in MODEL_ORDER}
    rank_sums = {m: 0.0 for m in MODEL_ORDER}
    per_comparison_rows = []

    for label in labels:
        finals = {m: all_results[m][label]["top2_accuracy"][-1] for m in MODEL_ORDER}
        best_val = max(finals.values())
        winners = [m for m in MODEL_ORDER if abs(finals[m] - best_val) < 1e-9]
        for m in winners:
            win_counts[m] += 1.0 / len(winners)

        # dense rank: 1 = best
        order = sorted(MODEL_ORDER, key=lambda m: -finals[m])
        ranks = {}
        rank = 1
        for m in order:
            ranks[m] = rank
            rank += 1
        for m in MODEL_ORDER:
            rank_sums[m] += ranks[m]

        per_comparison_rows.append((label, finals, ranks))

    mean_ranks = {m: rank_sums[m] / len(labels) for m in MODEL_ORDER}
    return win_counts, mean_ranks, per_comparison_rows


def plot_win_rate(win_counts, mean_ranks, n_comparisons):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color="#e1e0d9", linewidth=1, zorder=0)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)

    models_sorted_by_wins = sorted(MODEL_ORDER, key=lambda m: -win_counts[m])
    x = np.arange(len(models_sorted_by_wins))
    values = [win_counts[m] for m in models_sorted_by_wins]
    colors = [MODEL_STYLE[m]["color"] for m in models_sorted_by_wins]
    bars = ax1.bar(x, values, color=colors, width=0.55, zorder=2)
    for rect, val in zip(bars, values):
        ax1.text(rect.get_x() + rect.get_width() / 2, val + 0.3, f"{val:.1f}",
                  ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODEL_STYLE[m]["label"] for m in models_sorted_by_wins],
                         rotation=15, ha="right", fontsize=9, color=INK_SECONDARY)
    ax1.set_ylabel(f"Comparisons won (out of {n_comparisons})", fontsize=9.5, color=INK_SECONDARY)
    ax1.set_title("Best final top-2 accuracy, by model", fontsize=11.5, color=INK_PRIMARY, pad=8)

    models_sorted_by_rank = sorted(MODEL_ORDER, key=lambda m: mean_ranks[m])
    x2 = np.arange(len(models_sorted_by_rank))
    rank_values = [mean_ranks[m] for m in models_sorted_by_rank]
    rank_colors = [MODEL_STYLE[m]["color"] for m in models_sorted_by_rank]
    bars2 = ax2.bar(x2, rank_values, color=rank_colors, width=0.55, zorder=2)
    for rect, val in zip(bars2, rank_values):
        ax2.text(rect.get_x() + rect.get_width() / 2, val + 0.03, f"{val:.2f}",
                  ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([MODEL_STYLE[m]["label"] for m in models_sorted_by_rank],
                         rotation=15, ha="right", fontsize=9, color=INK_SECONDARY)
    ax2.set_ylabel("Mean rank (1 = best, 4 = worst)", fontsize=9.5, color=INK_SECONDARY)
    ax2.invert_yaxis()
    ax2.set_title("Average rank across all comparisons, by model", fontsize=11.5, color=INK_PRIMARY, pad=8)

    fig.suptitle(f"Overall model standing across {n_comparisons} comparisons (200 real target samples)",
                 fontsize=13, color=INK_PRIMARY, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "model_win_rate.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def write_csv(per_comparison_rows):
    out_path = os.path.join(ensure_out_dir(), "model_win_rate.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison"] + [f"{MODEL_STYLE[m]['label']}_accuracy" for m in MODEL_ORDER]
                         + [f"{MODEL_STYLE[m]['label']}_rank" for m in MODEL_ORDER])
        for label, finals, ranks in per_comparison_rows:
            writer.writerow([label] + [f"{finals[m]:.2f}" for m in MODEL_ORDER]
                             + [ranks[m] for m in MODEL_ORDER])
    print(f"Saved {out_path}")


if __name__ == "__main__":
    all_results = load_all_models()
    labels = all_comparison_labels(all_results)
    win_counts, mean_ranks, per_comparison_rows = compute_wins_and_ranks(all_results, labels)
    plot_win_rate(win_counts, mean_ranks, len(labels))
    write_csv(per_comparison_rows)
