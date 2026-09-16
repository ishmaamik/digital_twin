"""Plot clean colored mean recovery curves for College Avenue comparisons.

This is the colored-line style used in the thesis comparison figures. The
full-road Scenario 32 panel uses the older stored model files available for
that comparison; the straight-segment panel uses the consolidated 500-cap
files. The source files are kept explicit below because the two panels do
not have identical result-folder provenance.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

OUT_DIR = os.path.join("result", "diagrams", "cap500_study")
RANDOM_CHANCE = 12.5

MODEL_STYLE = {
    "mlp": {"label": "MLP", "color": "#2a78d6", "marker": "o", "linestyle": "--"},
    "knn": {"label": "k-NN", "color": "#eda100", "marker": "^", "linestyle": "-"},
    "rf": {"label": "Random Forest", "color": "#e87ba4", "marker": "D", "linestyle": "-"},
    "fourier_knn": {"label": "Fourier + k-NN", "color": "#008300", "marker": "v", "linestyle": "-"},
}

MODEL_ORDER = ["mlp", "knn", "rf", "fourier_knn"]

FILES = {
    "s1_to_s32": {
        "title": "McAllister day (narrow) -> College Ave day (narrow, bend included)",
        "files": {
            "mlp": os.path.join("result", "non-rehearsals", "stage1_s1_to_s32_acc.mat"),
            "knn": os.path.join("old knn", "stage1_s1_to_s32_rehearsal_knn_acc.mat"),
            "rf": os.path.join("old rf", "stage1_s1_to_s32_rehearsal_rf_acc.mat"),
            "fourier_knn": os.path.join("old knn", "stage1_s1_to_s32_rehearsal_fourier_knn_acc.mat"),
        },
    },
    "s1_to_s32straight": {
        "title": "McAllister day (narrow) -> College Ave day (narrow, straight segment only)",
        "files": {
            "mlp": os.path.join("result", "MLP", "stage1_s1_to_s32straight_rehearsal_acc.mat"),
            "knn": os.path.join("result", "new-knn-rf-fknn", "stage1_s1_to_s32straight_cap500_rehearsal_knn_acc.mat"),
            "rf": os.path.join("result", "new-knn-rf-fknn", "stage1_s1_to_s32straight_cap500_rehearsal_rf_acc.mat"),
            "fourier_knn": os.path.join("result", "new-knn-rf-fknn", "stage1_s1_to_s32straight_cap500_rehearsal_fourier_knn_acc.mat"),
        },
    },
}


def load_mean_curve(path):
    data = loadmat(path)
    points = data["sweep_points"].ravel()
    # acc: top-k metric x sweep point x seed
    values = np.asarray(data["acc"])[1] * 100.0
    return points, values.mean(axis=1)


def style_axes(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e1e0d9", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)


def plot_panel(ax, comparison):
    config = FILES[comparison]
    style_axes(ax)
    for model_name in MODEL_ORDER:
        path = config["files"][model_name]
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        points, mean = load_mean_curve(path)
        style = MODEL_STYLE[model_name]
        ax.plot(
            points,
            mean,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.3,
            markersize=4,
            label=style["label"],
        )
    ax.axhline(RANDOM_CHANCE, color="#898781", linestyle=":", linewidth=1.2,
               label="Random chance (12.5%)")
    ax.set_title(config["title"], fontsize=10.5, color="#0b0b0b", pad=8)
    ax.set_xlabel("Real target samples", fontsize=9, color="#52514e")
    ax.set_ylabel("Top-2 accuracy (%)", fontsize=9, color="#52514e")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 100)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, facecolor="#fcfcfb")
    plot_panel(axes[0], "s1_to_s32")
    plot_panel(axes[1], "s1_to_s32straight")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=5, frameon=False, fontsize=8.5, labelcolor="#0b0b0b")
    fig.suptitle(
        "College Avenue recovery comparison\n"
        "Top-2 accuracy under the available rehearsal result files",
        fontsize=13,
        color="#0b0b0b",
        y=1.055,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    os.makedirs(OUT_DIR, exist_ok=True)
    output = os.path.join(OUT_DIR, "college_ave_colored_curves_s32_vs_s32straight.png")
    fig.savefig(output, dpi=180, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()