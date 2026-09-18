"""Plot day-to-night and night-to-day 500-cap recovery comparisons.

Run from the repository root:
    python diagrams/cap500_study/plot_day_night_direction_comparisons.py
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    INK_MUTED,
    INK_PRIMARY,
    MODEL_ORDER,
    MODEL_STYLE,
    RANDOM_CHANCE_TOP2,
    SURFACE,
    ensure_out_dir,
    load_all_models,
)

GROUPS = {
    "mcallister": {
        "title": "McAllister Avenue: day/night transfer",
        "labels": ["s1_to_s2", "s2_to_s1"],
        "panel_titles": ["Day -> night", "Night -> day"],
        "filename": "day_night_mcallister_direction_comparison.png",
    },
    "ruralroad": {
        "title": "Rural Road: day/night transfer",
        "labels": ["s3_to_s4", "s4_to_s3"],
        "panel_titles": ["Day -> night", "Night -> day"],
        "filename": "day_night_rural_road_direction_comparison.png",
    },
}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e1e0d9", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)


def plot_group(group, results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, facecolor=SURFACE)
    for ax, label, panel_title in zip(axes, group["labels"], group["panel_titles"]):
        style_axes(ax)
        for model_name in MODEL_ORDER:
            style = MODEL_STYLE[model_name]
            curve = results[model_name][label]
            ax.plot(
                curve["points"],
                curve["top2_accuracy"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.4,
                markersize=5,
                label=style["label"],
            )
        ax.axhline(
            RANDOM_CHANCE_TOP2,
            color=INK_MUTED,
            linestyle=":",
            linewidth=1.2,
            label="Random chance (12.5%)",
        )
        ax.set_title(panel_title, fontsize=11, color=INK_PRIMARY, pad=8)
        ax.set_xlabel("Real target samples", fontsize=9.5, color="#52514e")
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 100)
        ax.set_xticks([0, 50, 100, 150, 200])

    axes[0].set_ylabel("Top-2 accuracy (%)", fontsize=9.5, color="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_PRIMARY,
    )
    fig.suptitle(
        f"{group['title']}\n"
        "500-cap synthetic-to-real-source rehearsal recovery",
        fontsize=13,
        color=INK_PRIMARY,
        y=1.055,
    )
    fig.text(
        0.5,
        0.01,
        "Source: synthetic Scenario 1 pretraining + real Scenario 1 rehearsal; target adaptation uses 500-cap protocol",
        ha="center",
        fontsize=8.5,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    output = os.path.join(ensure_out_dir(), group["filename"])
    fig.savefig(output, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    results = load_all_models()
    for group in GROUPS.values():
        plot_group(group, results)
