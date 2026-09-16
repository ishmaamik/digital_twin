"""
Shared loading, styling, and labeling helpers for diagrams/cap500_study/.

This is a DELIBERATELY FRESH loader -- it does not reuse diagrams/common.py's
MODEL_SUMMARY_FILES or load_model_summary(), which pointed at the older,
now-superseded result/summaries/*.json layout. Everything here reads from
three result folders found on 2026-09-17:

    result/MLP/                  -- k-NN/RF/Fourier-k-NN's sibling: the
                                     reference MLP, rehearsal-trained (source
                                     pool UNION N target samples, continued
                                     fine-tuning from the source checkpoint),
                                     raw per-comparison .mat files, all 32
                                     comparisons. Verified numerically
                                     identical to the older
                                     stage1_summary_rehearsal_base4/extended/
                                     batch3/batch4.json data (e.g. s1_to_s2
                                     final = 56.17% in both) -- this is a
                                     consolidated copy of that same run, not
                                     a new one.
    result/new-knn-rf-fknn/      -- k-NN, Random Forest, Fourier-k-NN,
                                     REHEARSAL PROTOCOL "rehearsal_source_cap_500":
                                     every comparison's source pool is capped
                                     at exactly 500 samples (not the full,
                                     scenario-size-dependent source pool used
                                     before), so every one of the 32
                                     comparisons is now trained on a MATCHED
                                     source budget. This supersedes the old
                                     varying-source-size knn/rf/fourier_knn
                                     data for these 32 comparisons.
    result/500-cap/              -- checked directly: byte-for-byte identical
                                     to the matching 4-comparison subset of
                                     result/new-knn-rf-fknn/ (confirmed via
                                     np.allclose on the raw arrays). This is
                                     an earlier, smaller pilot run that was
                                     later folded into the full 32-comparison
                                     new-knn-rf-fknn run; loading it
                                     separately would double-count nothing
                                     new, so it is not loaded here at all.

IMPORTANT ASYMMETRY, kept explicit rather than hidden: the MLP data uses a
FULL, uncapped source pool; the k-NN/RF/Fourier-k-NN data uses a 500-sample
CAPPED source pool. This is the same kind of protocol difference flagged
throughout this project's earlier diagrams -- never silently mixed, always
named in every caption that compares MLP against the other three.

Run nothing in this file directly; it is imported by the plot_*.py scripts.
"""
import json
import os
import re

import numpy as np
from scipy.io import loadmat

RESULT_DIR = "result"
MLP_DIR = os.path.join(RESULT_DIR, "MLP")
CAP500_DIR = os.path.join(RESULT_DIR, "new-knn-rf-fknn")
SYNTH_REAL_S32_DIR = os.path.join(RESULT_DIR, "synth_real_s32_rehearsal_500cap_small")
OUT_DIR = os.path.join(RESULT_DIR, "diagrams", "cap500_study")

MODEL_ORDER = ["mlp", "knn", "rf", "fourier_knn"]

MODEL_STYLE = {
    "mlp":         {"label": "MLP (rehearsal, full source)",  "color": "#2a78d6", "marker": "o", "linestyle": "--", "linewidth": 2.4, "zorder": 5},
    "knn":         {"label": "k-NN (500-source-cap)",          "color": "#eda100", "marker": "^", "linestyle": "-",  "linewidth": 2.0, "zorder": 3},
    "rf":          {"label": "Random Forest (500-source-cap)", "color": "#e87ba4", "marker": "D", "linestyle": "-",  "linewidth": 2.0, "zorder": 3},
    "fourier_knn": {"label": "Fourier + k-NN (500-source-cap)","color": "#008300", "marker": "v", "linestyle": "-",  "linewidth": 2.0, "zorder": 3},
}

SCENARIO_INFO = {
    "s1": "McAllister day (narrow)",
    "s2": "McAllister night (narrow)",
    "s3": "Rural Rd day (wide)",
    "s4": "Rural Rd night (wide)",
    "s7": "Site 7 day (wide)",
    "s32straight": "College Ave day (narrow)",
    "s33straight": "College Ave night (narrow)",
}

# The 10 comparisons individually narrated in the thesis (Stage 2's 4 +
# Stage 5's 6), used for the curated recovery-curve / bar / zero-shot /
# relative-power figures. The full 32-comparison matrix is handled
# separately (heatmap / win-rate / grand-average scripts).
STAGE2_LABELS = ["s1_to_s2", "s3_to_s4", "s1_to_s3", "s2_to_s4"]
STAGE2_NICE = {
    "s1_to_s2": "McAllister: day -> night",
    "s3_to_s4": "Rural Road: day -> night",
    "s1_to_s3": "Day: McAllister -> Rural Road",
    "s2_to_s4": "Night: McAllister -> Rural Road",
}
STAGE5_LABELS = [
    "s1_to_s7", "s1_to_s32straight", "s1_to_s33straight",
    "s2_to_s7", "s2_to_s32straight", "s2_to_s33straight",
]
STAGE5_NICE = {
    "s1_to_s7": "McAllister day -> Site 7",
    "s1_to_s32straight": "McAllister day -> College Ave day",
    "s1_to_s33straight": "McAllister day -> College Ave night",
    "s2_to_s7": "McAllister night -> Site 7",
    "s2_to_s32straight": "McAllister night -> College Ave day",
    "s2_to_s33straight": "McAllister night -> College Ave night",
}
CURATED_LABELS = STAGE2_LABELS + STAGE5_LABELS
CURATED_NICE = {**STAGE2_NICE, **STAGE5_NICE}

# Comparisons where straight-segment Scenario 32/33 is the TARGET: the
# fixed per-seed test partition can include one of a handful of raw NaN
# beam-power cells in those two scenarios, which silently NaNs the whole
# averaged relative-power statistic for that seed (never top-2 accuracy,
# which is a boolean hit/miss and never divides). Verified directly against
# both result/MLP and result/new-knn-rf-fknn on 2026-09-17: all 9 of these
# labels are 100% (MLP: partially, per-seed) NaN in top2_relative_power.
RELATIVE_POWER_NAN_LABELS = {
    "s1_to_s32straight", "s1_to_s33straight", "s2_to_s32straight", "s2_to_s33straight",
    "s3_to_s32straight", "s4_to_s33straight", "s7_to_s32straight",
    "s32straight_to_s33straight", "s33straight_to_s32straight",
}

RANDOM_CHANCE_TOP2 = 12.5

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
BASELINE_COLOR = "#c3c2b7"
SURFACE = "#fcfcfb"


def nice_label(label):
    src, tgt = label.split("_to_")
    return f"{SCENARIO_INFO.get(src, src)} -> {SCENARIO_INFO.get(tgt, tgt)}"


def _all_mlp_labels():
    labels = set()
    for fname in os.listdir(MLP_DIR):
        m = re.match(r"stage1_(.+)_rehearsal_acc\.mat", fname)
        if m:
            labels.add(m.group(1))
    return labels


def load_mlp():
    """{label: {"points": [...], "top2_accuracy": [...%], "top2_relative_power": [...%]}}
    Computed directly from the raw per-seed .mat arrays (shape: 4 top-k x 23
    sweep points x 10 seeds), mean over the seed axis, top-2 = index 1.
    Relative power uses np.nanmean per sweep point (only where not fully
    NaN) so a single bad seed doesn't need special-casing beyond what
    RELATIVE_POWER_NAN_LABELS already documents as unusable."""
    out = {}
    for label in _all_mlp_labels():
        acc = loadmat(os.path.join(MLP_DIR, f"stage1_{label}_rehearsal_acc.mat"))
        pwr = loadmat(os.path.join(MLP_DIR, f"stage1_{label}_rehearsal_pwr.mat"))
        points = acc["sweep_points"].ravel().tolist()
        acc_top2 = acc["acc"][1]  # (23, 10)
        pwr_top2 = pwr["pwr"][1]
        out[label] = {
            "points": points,
            "top2_accuracy": (acc_top2.mean(axis=1) * 100).tolist(),
            "top2_relative_power": (np.nanmean(pwr_top2, axis=1) * 100).tolist(),
        }
    return out


def load_cap500_model(model_name):
    """knn / rf / fourier_knn, from the single aggregated JSON summary."""
    path = os.path.join(CAP500_DIR, f"stage1_summary_{model_name}_rehearsal.json")
    with open(path) as f:
        d = json.load(f)
    out = {}
    for label, vals in d["comparisons"].items():
        out[label] = {
            "points": d["sweep_points"],
            "top2_accuracy": [v * 100 for v in vals["top2_accuracy"]],
            "top2_relative_power": [v * 100 for v in vals["top2_relative_power"]],
        }
    return out


def load_all_models():
    return {
        "mlp": load_mlp(),
        "knn": load_cap500_model("knn"),
        "rf": load_cap500_model("rf"),
        "fourier_knn": load_cap500_model("fourier_knn"),
    }


def all_comparison_labels(all_results):
    sets = [set(all_results[m].keys()) for m in MODEL_ORDER]
    return sorted(set.intersection(*sets))


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_COLOR, linewidth=1, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE_COLOR)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR
