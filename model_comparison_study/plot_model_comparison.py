"""
Model comparison figures: every model from the Stage 1 architecture study
(model_comparison_study/) plotted against the original MLP, in the same
"accuracy vs. real target samples" style as the original recovery-curve
figure (maltab/plot_stage1_comparison.m), but pure Python -- no MATLAB.

Reads whatever result files are already on disk:
  - result/stage1_summary.json                  (original mlp)
  - result/stage1_summary_<model>.json           (a model that finished all
                                                    4 comparisons)
  - result/stage1_<comparison>_<model>_acc.mat   (fallback for a model that
                                                    only finished some
                                                    comparisons, e.g. a
                                                    partial/interrupted run)
A model with no results at all (e.g. one never run) is silently skipped, so
this script can be re-run as more models finish without editing anything.

Produces:
  result/model_comparison_recovery_curves.png   -- 2x2 grid, one panel per
                                                    comparison, one line per
                                                    model
  result/model_comparison_final_accuracy.png    -- 2x2 grid of bar charts,
                                                    each model's accuracy at
                                                    the largest common real-
                                                    sample count

Must be run from the repository root:
    python model_comparison_study/plot_model_comparison.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

RESULT_DIR = "result"
COMPARISONS = [
    ("time_of_day_at_mcallister", "McAllister: day -> night"),
    ("time_of_day_at_ruralroad", "Rural Road: day -> night"),
    ("site_during_day", "Day: McAllister -> Rural Road"),
    ("site_during_night", "Night: McAllister -> Rural Road"),
]

# Fixed categorical order (dataviz skill palette, light mode) -- assigned by
# slot, never re-cycled, so a model's color/marker stays fixed across every
# figure regardless of which subset of models has data in a given panel.
MODEL_STYLE = {
    "mlp":            {"label": "MLP (original)",        "color": "#2a78d6", "marker": "o", "linestyle": "--", "linewidth": 2.4, "zorder": 5},
    "tinymlp":        {"label": "TinyMLP",                "color": "#eb6834", "marker": "s", "linestyle": "-",  "linewidth": 2.0, "zorder": 3},
    "resmlp":         {"label": "ResMLP",                 "color": "#1baf7a", "marker": "*", "linestyle": "-",  "linewidth": 2.0, "zorder": 3},
    "knn":            {"label": "k-NN",                   "color": "#eda100", "marker": "^", "linestyle": "-",  "linewidth": 2.0, "zorder": 3},
    "rf":             {"label": "Random Forest",          "color": "#e87ba4", "marker": "D", "linestyle": "-",  "linewidth": 2.0, "zorder": 3},
    "fourier_knn":    {"label": "Fourier + k-NN",         "color": "#008300", "marker": "v", "linestyle": "-",  "linewidth": 2.0, "zorder": 3},
    "fourier_rf":     {"label": "Fourier + Random Forest","color": "#4a3aa7", "marker": "P", "linestyle": "-",  "linewidth": 2.0, "zorder": 3},
    "fttransformer":  {"label": "FT-Transformer",         "color": "#e34948", "marker": "X", "linestyle": "-",  "linewidth": 2.0, "zorder": 3},
}
MODEL_ORDER = list(MODEL_STYLE.keys())

# Chart chrome (dataviz skill reference palette, light mode)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
BASELINE_COLOR = "#c3c2b7"
SURFACE = "#fcfcfb"

RANDOM_CHANCE_TOP2 = 12.5  # 2 of 16 beams


def load_model_results(model_name):
    """Returns {comparison_label: {"points": [...], "top2_accuracy": [...] (0-100)}}
    for whichever comparisons this model has data for."""
    summary_path = os.path.join(RESULT_DIR, "stage1_summary.json" if model_name == "mlp"
                                 else f"stage1_summary_{model_name}.json")
    results = {}

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            d = json.load(f)
        points = d["sweep_points"]
        for label, vals in d["comparisons"].items():
            results[label] = {
                "points": points,
                "top2_accuracy": [v * 100 for v in vals["top2_accuracy"]],
            }
        return results

    # Fallback: a model that never finished all 4 comparisons (its combined
    # summary JSON was never written) -- read whatever raw per-comparison
    # .mat files do exist.
    for label, _ in COMPARISONS:
        mat_path = os.path.join(RESULT_DIR, f"stage1_{label}_{model_name}_acc.mat")
        if not os.path.exists(mat_path):
            continue
        d = loadmat(mat_path)
        acc = d["acc"]  # (4 metrics, points, seeds)
        points = d["sweep_points"].flatten().tolist()
        top2 = acc[1].mean(axis=-1) * 100  # mean over seeds
        results[label] = {"points": points, "top2_accuracy": top2.tolist()}
    return results


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_COLOR, linewidth=1, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE_COLOR)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_recovery_curves(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), facecolor=SURFACE)
    fig.suptitle("Recovery curves by model: top-2 beam accuracy vs. real target samples",
                 fontsize=14, color=INK_PRIMARY, y=0.985)

    present_models = []
    for ax, (label, nice_name) in zip(axes.flat, COMPARISONS):
        style_axes(ax)
        ax.set_title(nice_name, fontsize=11, color=INK_PRIMARY, pad=8)
        ax.axhline(RANDOM_CHANCE_TOP2, color=INK_MUTED, linestyle=":", linewidth=1.2, zorder=1)

        for model_name in MODEL_ORDER:
            data = all_results.get(model_name, {}).get(label)
            if data is None:
                continue
            style = MODEL_STYLE[model_name]
            ax.plot(data["points"], data["top2_accuracy"],
                     color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                     linewidth=style["linewidth"], markersize=7, markeredgecolor=SURFACE,
                     markeredgewidth=1, zorder=style["zorder"])
            if model_name not in present_models:
                present_models.append(model_name)

        ax.set_xlabel("Real target-domain samples", fontsize=9, color=INK_SECONDARY)
        ax.set_ylabel("Top-2 accuracy (%)", fontsize=9, color=INK_SECONDARY)
        ax.set_ylim(0, 100)

    # One shared legend for the whole figure, ordered by the fixed slot
    # order (never re-cycled), listing only models that actually appear.
    handles = [plt.Line2D([0], [0], color=MODEL_STYLE[m]["color"], marker=MODEL_STYLE[m]["marker"],
                           linestyle=MODEL_STYLE[m]["linestyle"], linewidth=MODEL_STYLE[m]["linewidth"],
                           markersize=7, label=MODEL_STYLE[m]["label"])
               for m in present_models]
    handles.append(plt.Line2D([0], [0], color=INK_MUTED, linestyle=":", linewidth=1.2,
                               label="Random chance (top-2 of 16 beams)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    out_path = os.path.join(RESULT_DIR, "model_comparison_recovery_curves.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_final_accuracy_bars(all_results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=SURFACE)
    fig.suptitle("Top-2 accuracy at 200 real target samples, by model",
                 fontsize=14, color=INK_PRIMARY, y=0.98)

    for ax, (label, nice_name) in zip(axes.flat, COMPARISONS):
        style_axes(ax)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=1, zorder=0)
        ax.grid(axis="x", visible=False)
        ax.set_title(nice_name, fontsize=11, color=INK_PRIMARY, pad=8)
        ax.axhline(RANDOM_CHANCE_TOP2, color=INK_MUTED, linestyle=":", linewidth=1.2, zorder=1)

        bar_models, bar_values, bar_colors = [], [], []
        for model_name in MODEL_ORDER:
            data = all_results.get(model_name, {}).get(label)
            if data is None:
                continue
            # use the largest sample count this model actually has for this comparison
            idx = len(data["points"]) - 1
            bar_models.append(MODEL_STYLE[model_name]["label"])
            bar_values.append(data["top2_accuracy"][idx])
            bar_colors.append(MODEL_STYLE[model_name]["color"])

        order = np.argsort(bar_values)[::-1]
        bar_models = [bar_models[i] for i in order]
        bar_values = [bar_values[i] for i in order]
        bar_colors = [bar_colors[i] for i in order]

        x = np.arange(len(bar_models))
        bars = ax.bar(x, bar_values, color=bar_colors, width=0.6, zorder=2)
        for rect, val in zip(bars, bar_values):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 1.5, f"{val:.1f}%",
                     ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)

        ax.set_xticks(x)
        ax.set_xticklabels(bar_models, rotation=35, ha="right", fontsize=8, color=INK_SECONDARY)
        ax.set_ylabel("Top-2 accuracy (%)", fontsize=9, color=INK_SECONDARY)
        ax.set_ylim(0, 100)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(RESULT_DIR, "model_comparison_final_accuracy.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    all_results = {m: load_model_results(m) for m in MODEL_ORDER}
    for m in MODEL_ORDER:
        n = len(all_results[m])
        print(f"{m:16s} data for {n}/4 comparisons")

    plot_recovery_curves(all_results)
    plot_final_accuracy_bars(all_results)
