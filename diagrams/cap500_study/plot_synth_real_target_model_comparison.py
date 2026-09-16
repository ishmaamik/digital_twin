"""Plot synthetic -> real Scenario 1 -> real target model comparisons.

Run from the repository root:
    python diagrams/cap500_study/plot_synth_real_target_model_comparison.py
"""
import csv
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
    "scenario2": "Scenario 2: McAllister night",
    "scenario3": "Scenario 3: Rural Road day",
    "scenario7": "Scenario 7: wide 4-lane site day",
    "scenario33straight": "Scenario 33straight: College Avenue night, straight segment",
    "scenario33": "Scenario 33: College Avenue night, full road",
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
            f"stage2_synth_real_{target.replace('scenario', 's')}_rehearsal_500cap_summary.json",
        )
    else:
        folder = f"synth_real_target_nonparametric_500cap_{model}"
        filename = f"stage2_{model}_synth_real1_{target}_rehearsal_500cap_summary.json"
        path = os.path.join(RESULT_ROOT, folder, filename)
    return np.asarray(load_json(path)["top2_accuracy"]) * 100.0


def load_summary(model, target):
    if model == "mlp":
        folder = f"synth_real_{target.replace('scenario', 's')}_rehearsal_500cap_small"
        filename = f"stage2_synth_real_{target.replace('scenario', 's')}_rehearsal_500cap_summary.json"
    else:
        folder = f"synth_real_target_nonparametric_500cap_{model}"
        filename = f"stage2_{model}_synth_real1_{target}_rehearsal_500cap_summary.json"
    return load_json(os.path.join(RESULT_ROOT, folder, filename))


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
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharey=True, facecolor="#fcfcfb")
    axes = axes.ravel()
    table_rows = []
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
            summary = load_summary(model, target)
            table_rows.append({
                "target": target,
                "target_label": TARGET_LABELS[target],
                "model": model,
                "zero_shot_top2_percent": float(summary["top2_accuracy"][0] * 100),
                "at_200_target_top2_percent": float(summary["top2_accuracy"][-1] * 100),
                "zero_shot_relative_power_percent": float(summary["top2_relative_power"][0] * 100),
                "at_200_target_relative_power_percent": float(summary["top2_relative_power"][-1] * 100),
            })
        ax.axhline(RANDOM_CHANCE, color="#898781", linestyle=":", linewidth=1.2,
                   label="Random chance (12.5%)")
        ax.set_title(TARGET_LABELS[target], fontsize=10.5, color="#0b0b0b", pad=8)
        ax.set_xlabel("Real target samples", fontsize=9, color="#52514e")
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 100)
        ax.set_xticks(POINTS)
    axes[0].set_ylabel("Top-2 accuracy (%)", fontsize=9.5, color="#52514e")
    axes[3].set_ylabel("Top-2 accuracy (%)", fontsize=9.5, color="#52514e")
    axes[-1].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=5, frameon=False, fontsize=8.5, labelcolor="#0b0b0b")
    fig.suptitle(
        "Synthetic Scenario 1 pretraining -> real Scenario 1 rehearsal -> real target deployment\n"
        "MLP versus non-parametric models across five target scenarios",
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
    output = os.path.join(OUT_DIR, "synth_real_target_model_comparison_five_targets.png")
    fig.savefig(output, dpi=180, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")

    # Write one separate graph per target for thesis insertion when a single
    # comparison needs to be shown at larger size.
    for target in TARGET_LABELS:
        fig, ax = plt.subplots(figsize=(9, 6), facecolor="#fcfcfb")
        style_axes(ax)
        for model, style in MODELS.items():
            ax.plot(POINTS, load_curve(model, target), color=style["color"],
                    marker=style["marker"], linestyle=style["linestyle"],
                    linewidth=2.4, markersize=5, label=style["label"])
        ax.axhline(RANDOM_CHANCE, color="#898781", linestyle=":", linewidth=1.2,
                   label="Random chance (12.5%)")
        ax.set_title(TARGET_LABELS[target] + "\n500-cap rehearsal recovery", fontsize=12.5,
                     color="#0b0b0b", pad=12)
        ax.set_xlabel("Real target samples", fontsize=10, color="#52514e")
        ax.set_ylabel("Top-2 accuracy (%)", fontsize=10, color="#52514e")
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 100)
        ax.set_xticks(POINTS)
        ax.legend(frameon=False, fontsize=9, labelcolor="#52514e")
        fig.tight_layout()
        single_output = os.path.join(OUT_DIR, f"synth_real_target_{target}_four_model_accuracy.png")
        fig.savefig(single_output, dpi=180, facecolor="#fcfcfb", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {single_output}")

    columns = list(table_rows[0])
    csv_output = os.path.join(OUT_DIR, "synth_real_target_five_comparison_summary.csv")
    with open(csv_output, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Saved {csv_output}")

    markdown_output = os.path.join(OUT_DIR, "synth_real_target_five_comparison_summary.md")
    with open(markdown_output, "w", encoding="utf-8") as file:
        file.write("# Synthetic to Real-Target 500-Cap Summary\n\n")
        file.write("Protocol: full synthetic Scenario 1 pretraining, then 250 synthetic + 250 real Scenario 1 replay rows, plus the listed number of real target rows. Results are averaged over 3 seeds.\n\n")
        file.write("| Target | Model | 0 target samples (%) | 200 target samples (%) | Gain (percentage points) |\n|---|---|---:|---:|---:|\n")
        for row in table_rows:
            gain = row["at_200_target_top2_percent"] - row["zero_shot_top2_percent"]
            file.write(f"| {row['target_label']} | {MODELS[row['model']]['label']} | {row['zero_shot_top2_percent']:.2f} | {row['at_200_target_top2_percent']:.2f} | {gain:+.2f} |\n")
    print(f"Saved {markdown_output}")


if __name__ == "__main__":
    main()