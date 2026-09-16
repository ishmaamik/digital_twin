"""Plot synthetic -> real Scenario 1 -> real target model comparisons."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULT_ROOT = "result"
OUT_DIR = os.path.join(RESULT_ROOT, "diagrams", "cap500_study")
RANDOM_CHANCE = 12.5
POINTS = [0, 50, 100, 150, 200]

MODELS = {
    "mlp": {"label": "MLP", "color": "#2a78d6", "marker": "o", "linestyle": "--"},
    "rf": {"label": "Random Forest", "color": "#e87ba4", "marker": "D", "linestyle": "-"},
    "knn": {"label": "k-NN", "color": "#eda100", "marker": "^", "linestyle": "-"},
    "fourier_knn": {"label": "Fourier + k-NN", "color": "#008300", "marker": "v", "linestyle": "-"},
}

TARGET_LABELS = {
    "scenario32": "Scenario 32: full College Avenue road (bend included)",
    "scenario32straight": "Scenario 32straight: straight segment only",
}


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def load_curve(model, target):
    if model == "mlp":
        folder = f"synth_real_{target.replace('scenario', 's')}_rehearsal_500cap_small"
        path = os.path.join(
            RESULT_ROOT,
            folder,
            "stage2_synth_real_s32_rehearsal_500cap_summary.json",
        )
    else:
        folder = f"synth_real_target_nonparametric_500cap_{model}"
        filename = f"stage2_{model}_synth_real1_{target}_rehearsal_500cap_summary.json"
        path = os.path.join(RESULT_ROOT, folder, filename)
    return np.asarray(load_json(path)["top2_accuracy"]) * 100.0


def style_axes(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e1e0d9", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, facecolor="#fcfcfb")
    for ax, target in zip(axes, TARGET_LABELS):
        style_axes(ax)
        for model, style in MODELS.items():
            accuracy = load_curve(model, target)
            ax.plot(
                POINTS,
                accuracy,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.4,
                markersize=5,
                label=style["label"],
            )
        ax.axhline(RANDOM_CHANCE, color="#898781", linestyle=":", linewidth=1.2,
                   label="Random chance (12.5%)")
        ax.set_title(TARGET_LABELS[target], fontsize=10.5, color="#0b0b0b", pad=8)
        ax.set_xlabel("Real target samples", fontsize=9, color="#52514e")
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 100)
        ax.set_xticks(POINTS)
    axes[0].set_ylabel("Top-2 accuracy (%)", fontsize=9.5, color="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=5, frameon=False, fontsize=8.5, labelcolor="#0b0b0b")
    fig.suptitle(
        "Synthetic pretraining -> real Scenario 1 rehearsal -> real target deployment\n"
        "MLP versus non-parametric models",
        fontsize=13,
        color="#0b0b0b",
        y=1.055,
    )
    fig.text(
        0.5,
        0.01,
        "At N > 0: 250 synthetic Scenario 1 + 250 real Scenario 1 replay rows + N real target rows; 3 seeds",
        ha="center",
        fontsize=8.5,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    output = os.path.join(OUT_DIR, "synth_real_target_model_comparison.png")
    fig.savefig(output, dpi=180, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()