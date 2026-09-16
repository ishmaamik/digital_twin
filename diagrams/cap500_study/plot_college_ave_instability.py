"""Plot per-seed instability for the two College Avenue MLP comparisons.

The two inputs are the original non-rehearsal accuracy results:
    result/non-rehearsals/stage1_s1_to_s32_acc.mat
    result/non-rehearsals/stage1_s1_to_s32straight_acc.mat

Run from the repository root:
    python diagrams/cap500_study/plot_college_ave_instability.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

OUT_DIR = os.path.join("result", "diagrams", "cap500_study")
RANDOM_CHANCE = 12.5

COMPARISONS = {
    "s1_to_s32": {
        "path": os.path.join("result", "non-rehearsals", "stage1_s1_to_s32_acc.mat"),
        "title": "McAllister day -> College Avenue day (full road, bend included)",
        "filename": "college_ave_s1_to_s32_instability.png",
    },
    "s1_to_s32straight": {
        "path": os.path.join("result", "non-rehearsals", "stage1_s1_to_s32straight_acc.mat"),
        "title": "McAllister day -> College Avenue day (straight segment only)",
        "filename": "college_ave_s1_to_s32straight_instability.png",
    },
}


def load_top2(path):
    data = loadmat(path)
    points = data["sweep_points"].ravel()
    top2 = np.asarray(data["acc"])[1] * 100.0
    return points, top2


def style_axis(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e1e0d9", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)


def plot_one(key, config):
    points, seed_accuracy = load_top2(config["path"])
    mean = seed_accuracy.mean(axis=1)
    std = seed_accuracy.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="#fcfcfb")
    style_axis(ax)

    for seed_idx in range(seed_accuracy.shape[1]):
        ax.plot(
            points,
            seed_accuracy[:, seed_idx],
            color="#8c9bab",
            linewidth=1.0,
            alpha=0.38,
            marker=".",
            markersize=3,
            label="Individual seed" if seed_idx == 0 else None,
        )

    ax.fill_between(
        points,
        mean - std,
        mean + std,
        color="#2a78d6",
        alpha=0.16,
        label="Mean +/- 1 standard deviation",
    )
    ax.plot(
        points,
        mean,
        color="#2a78d6",
        linewidth=2.8,
        marker="o",
        markersize=4.5,
        label="Mean across 10 seeds",
        zorder=4,
    )
    ax.axhline(
        RANDOM_CHANCE,
        color="#898781",
        linestyle=":",
        linewidth=1.3,
        label="Random chance (12.5%)",
    )

    final_spread = seed_accuracy[-1].max() - seed_accuracy[-1].min()
    ax.set_title(
        f"College Avenue instability: {config['title']}\n"
        f"Top-2 accuracy across 10 random seeds | final spread: {final_spread:.1f} percentage points",
        fontsize=12.5,
        color="#0b0b0b",
        pad=12,
    )
    ax.set_xlabel("Real target samples", fontsize=10, color="#52514e")
    ax.set_ylabel("Top-2 accuracy (%)", fontsize=10, color="#52514e")
    ax.set_xlim(points.min(), points.max())
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#52514e")

    os.makedirs(OUT_DIR, exist_ok=True)
    output = os.path.join(OUT_DIR, config["filename"])
    fig.tight_layout()
    fig.savefig(output, dpi=180, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    print(
        f"  {key}: final mean={mean[-1]:.2f}%, std={std[-1]:.2f}, "
        f"min={seed_accuracy[-1].min():.2f}%, max={seed_accuracy[-1].max():.2f}%"
    )


def plot_side_by_side():
    loaded = {key: load_top2(config["path"]) for key, config in COMPARISONS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True, facecolor="#fcfcfb")

    for ax, (key, config) in zip(axes, COMPARISONS.items()):
        points, seed_accuracy = loaded[key]
        mean = seed_accuracy.mean(axis=1)
        std = seed_accuracy.std(axis=1)
        style_axis(ax)
        for seed_idx in range(seed_accuracy.shape[1]):
            ax.plot(points, seed_accuracy[:, seed_idx], color="#8c9bab", alpha=0.38, linewidth=0.9)
        ax.fill_between(points, mean - std, mean + std, color="#2a78d6", alpha=0.16)
        ax.plot(points, mean, color="#2a78d6", linewidth=2.5, marker="o", markersize=3.5)
        ax.axhline(RANDOM_CHANCE, color="#898781", linestyle=":", linewidth=1.2)
        ax.set_title(config["title"], fontsize=10.5, color="#0b0b0b")
        ax.set_xlabel("Real target samples", fontsize=9, color="#52514e")
        ax.set_xlim(points.min(), points.max())
        ax.set_ylim(0, 100)

    axes[0].set_ylabel("Top-2 accuracy (%)", fontsize=9.5, color="#52514e")
    fig.suptitle(
        "College Avenue instability comparison\n"
        "Per-seed top-2 accuracy: bend included versus straight segment only",
        fontsize=13,
        color="#0b0b0b",
        y=1.03,
    )
    fig.text(0.5, 0.01, "Gray lines: individual seeds | Blue line: mean | Blue band: +/- 1 standard deviation | Dotted line: 12.5% random chance",
             ha="center", fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    output = os.path.join(OUT_DIR, "college_ave_instability_side_by_side.png")
    fig.savefig(output, dpi=180, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    for comparison_key, comparison_config in COMPARISONS.items():
        plot_one(comparison_key, comparison_config)
    plot_side_by_side()