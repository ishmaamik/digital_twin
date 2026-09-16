"""Plot synthetic-pretraining plus real-source rehearsal deployment failure.

Inputs:
  result/synth_real_s32_rehearsal_500cap_small/
  result/synth_real_s32straight_rehearsal_500cap_small/

The stored summaries contain top-2 accuracy at target sweep points 0, 50,
100, 150, and 200. Point 0 is the synthetic-only zero-shot evaluation. The
remaining points use the 500-sample source replay pool (250 synthetic plus
250 real Scenario 1) together with N real target samples.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULT_ROOT = os.path.join("result")
OUT_DIR = os.path.join(RESULT_ROOT, "diagrams", "cap500_study")
RANDOM_CHANCE = 12.5

EXPERIMENTS = {
    "s32": {
        "folder": "synth_real_s32_rehearsal_500cap_small",
        "label": "Scenario 32: College Avenue full road\n(bend included)",
        "color": "#d95f02",
        "filename": "synth_real_target_failure_s32.png",
    },
    "s32straight": {
        "folder": "synth_real_s32straight_rehearsal_500cap_small",
        "label": "Scenario 32straight: College Avenue\n(straight segment only)",
        "color": "#1b6ca8",
        "filename": "synth_real_target_failure_s32straight.png",
    },
}


def load_summary(config):
    path = os.path.join(
        RESULT_ROOT,
        config["folder"],
        "stage2_synth_real_s32_rehearsal_500cap_summary.json",
    )
    with open(path, encoding="utf-8") as file:
        summary = json.load(file)
    return np.asarray(summary["sweep_points"], dtype=float), np.asarray(summary["top2_accuracy"], dtype=float) * 100.0


def style_axes(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e1e0d9", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)


def draw_curve(ax, config, points, accuracy):
    color = config["color"]
    style_axes(ax)
    ax.axhline(RANDOM_CHANCE, color="#898781", linestyle=":", linewidth=1.3,
               label="Random chance (12.5%)")
    ax.plot(points[0], accuracy[0], marker="o", markersize=9, color="#111111",
            markerfacecolor="#ffffff", markeredgewidth=2, zorder=5,
            label="Synthetic-only zero-shot")
    ax.plot(points[1:], accuracy[1:], color=color, linewidth=2.8, marker="o",
            markersize=6, label="500 source replay + real target samples", zorder=4)
    ax.plot(points, accuracy, color=color, linewidth=1.0, alpha=0.25, zorder=2)
    ax.annotate(
        f"{accuracy[0]:.1f}%",
        (points[0], accuracy[0]),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
        color="#111111",
    )
    ax.annotate(
        f"{accuracy[-1]:.1f}%",
        (points[-1], accuracy[-1]),
        xytext=(-38, 10),
        textcoords="offset points",
        fontsize=9,
        color=color,
    )
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)
    ax.set_xticks(points)
    ax.set_xlabel("Real target samples used for deployment adaptation", fontsize=9, color="#52514e")
    ax.set_ylabel("Top-2 accuracy (%)", fontsize=9, color="#52514e")


def main():
    loaded = {key: load_summary(config) for key, config in EXPERIMENTS.items()}
    os.makedirs(OUT_DIR, exist_ok=True)

    for key, config in EXPERIMENTS.items():
        points, accuracy = loaded[key]
        fig, ax = plt.subplots(figsize=(9, 6), facecolor="#fcfcfb")
        draw_curve(ax, config, points, accuracy)
        ax.set_title(
            f"Synthetic + real source rehearsal fails on real-target deployment\n{config['label']}",
            fontsize=13,
            color="#0b0b0b",
            pad=12,
        )
        ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#52514e")
        fig.tight_layout()
        output = os.path.join(OUT_DIR, config["filename"])
        fig.savefig(output, dpi=180, facecolor="#fcfcfb", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, facecolor="#fcfcfb")
    for ax, (key, config) in zip(axes, EXPERIMENTS.items()):
        points, accuracy = loaded[key]
        draw_curve(ax, config, points, accuracy)
        ax.set_title(config["label"], fontsize=11, color="#0b0b0b", pad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=3, frameon=False, fontsize=9, labelcolor="#0b0b0b")
    fig.suptitle(
        "Synthetic + real source rehearsal versus real-target deployment\n"
        "The target-site model remains close to the random baseline",
        fontsize=13,
        color="#0b0b0b",
        y=1.055,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output = os.path.join(OUT_DIR, "synth_real_target_failure_s32_vs_s32straight.png")
    fig.savefig(output, dpi=180, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()