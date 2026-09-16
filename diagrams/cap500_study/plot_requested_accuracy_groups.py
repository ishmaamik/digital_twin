"""Plot separate top-2 accuracy recovery diagrams for requested site groups.

Run from the repository root:
    python diagrams/cap500_study/plot_requested_accuracy_groups.py

The figures use the consolidated 500-cap comparison results. MLP results are
the repository's rehearsal MLP results, while the other models use the
matched 500-source-cap results loaded by common.py.
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
    all_comparison_labels,
    ensure_out_dir,
    load_all_models,
    nice_label,
    style_axes,
)


NARROW = {"s1", "s2", "s32straight", "s33straight"}
SIX_LANE_WIDE = {"s3", "s4"}


def pair_parts(label):
    return label.split("_to_")


def select_labels(all_labels):
    groups = {
        "narrow_to_narrow": [
            label for label in all_labels
            if pair_parts(label)[0] in NARROW and pair_parts(label)[1] in NARROW
        ],
        "narrow_to_wide_6_lanes": [
            label for label in all_labels
            if pair_parts(label)[0] in NARROW and pair_parts(label)[1] in SIX_LANE_WIDE
        ],
        # Scenario 7 is the four-lane wide site. The requested wording says
        # "narrow 4 lanes"; the title makes the actual geometry explicit.
        "narrow_2_lanes_to_wide_4_lanes": [
            label for label in all_labels
            if pair_parts(label)[0] in {"s1", "s2"} and pair_parts(label)[1] == "s7"
        ],
        "narrow_day_to_night": ["s1_to_s2", "s32straight_to_s33straight"],
        "narrow_day_to_narrow_day": ["s1_to_s32straight", "s32straight_to_s1"],
        "wide_day_to_wide_night": ["s3_to_s4"],
    }
    return groups


def plot_group(group_key, labels, results):
    ncols = 2 if len(labels) > 2 else len(labels)
    ncols = max(ncols, 1)
    nrows = (len(labels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2 * ncols, 5.1 * nrows),
        squeeze=False,
        facecolor=SURFACE,
    )
    axes = axes.ravel()

    for ax, label in zip(axes, labels):
        style_axes(ax)
        for model_name in MODEL_ORDER:
            style = MODEL_STYLE[model_name]
            data = results[model_name][label]
            ax.plot(
                data["points"],
                data["top2_accuracy"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=4,
                label=style["label"],
                zorder=style["zorder"],
            )
        ax.axhline(
            RANDOM_CHANCE_TOP2,
            color=INK_MUTED,
            linestyle=":",
            linewidth=1.2,
            label="Random chance (12.5%)",
            zorder=1,
        )
        ax.set_title(nice_label(label), fontsize=10.5, color=INK_PRIMARY, pad=8)
        ax.set_xlabel("Real target samples", fontsize=9, color=INK_PRIMARY)
        ax.set_ylabel("Top-2 accuracy (%)", fontsize=9, color=INK_PRIMARY)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 100)

    for ax in axes[len(labels):]:
        ax.set_visible(False)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_PRIMARY,
    )
    titles = {
        "narrow_to_narrow": "Narrow site to narrow site",
        "narrow_to_wide_6_lanes": "Narrow site to wide 6-lane site",
        "narrow_2_lanes_to_wide_4_lanes": "Narrow 2-lane site to wide 4-lane site",
        "narrow_day_to_night": "Narrow site: day to night",
        "narrow_day_to_narrow_day": "Narrow site: day to day",
        "wide_day_to_wide_night": "Wide site: day to night",
    }
    fig.suptitle(
        f"{titles[group_key]}\n"
        "Top-2 accuracy recovery under the 500-source-cap rehearsal study",
        fontsize=13,
        color=INK_PRIMARY,
        y=1.045,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    filename = f"requested_{group_key}_accuracy.png"
    if group_key == "wide_day_to_wide_night":
        filename = "requested_wide_day_to_wide_night_accuracy_v2.png"
    out_path = os.path.join(ensure_out_dir(), filename)
    fig.savefig(out_path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path} ({len(labels)} comparisons)")


def main():
    results = load_all_models()
    labels = all_comparison_labels(results)
    groups = select_labels(labels)
    for group_key, group_labels in groups.items():
        if not group_labels:
            raise ValueError(f"No available results for {group_key}")
        plot_group(group_key, group_labels, results)


if __name__ == "__main__":
    main()