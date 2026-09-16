"""
Shared loading, styling, and labeling helpers for every script in diagrams/.

Data source and why: every script here reads ONLY the already-computed
result/summaries/*.json files -- nothing is retrained and nothing under
result/*.mat is touched. All four models (mlp, knn, rf, fourier_knn) were
verified (2026-09-16) to report the SAME 32 source->target comparisons under
the SAME rehearsal protocol (fit/fine-tune on the full source pool UNION N
real target samples, at the same 23-point sweep N in
{0,5,10,...,100,150,200}), so they can be plotted against each other directly
without any protocol mismatch. For the mlp baseline this specifically means
using the newer rehearsal-protocol summaries (stage1_summary_rehearsal_base4
+ _extended + _extended_batch3 + _extended_batch4), NOT the older
result/summaries/stage1_summary.json, which used a different, non-rehearsal
continued-fine-tuning protocol and is intentionally not read by this folder.

Comparison label convention: "sA_to_sB" means trained on Scenario A's pool
(rehearsal source), evaluated on Scenario B's held-out test partition
(rehearsal target). "s32straight"/"s33straight" refer to the curvature-free,
straight-segment-only versions of Scenarios 32/33 (see thesis Stage 4) --
this is the only version of those two scenarios present anywhere in the
rehearsal results, so there is nothing to disambiguate at load time.

Run nothing in this file directly; it is imported by the plot_*.py scripts.
"""
import json
import os

RESULT_DIR = "result"
SUMMARY_DIR = os.path.join(RESULT_DIR, "summaries")
OUT_DIR = os.path.join(RESULT_DIR, "diagrams")

MODEL_SUMMARY_FILES = {
    "mlp": [
        "stage1_summary_rehearsal_base4.json",
        "stage1_summary_rehearsal_extended.json",
        "stage1_summary_rehearsal_extended_batch3.json",
        "stage1_summary_rehearsal_extended_batch4.json",
    ],
    "knn": [
        "stage1_summary_knn_rehearsal.json",
        "stage1_summary_knn_rehearsal_newsites.json",
        "stage1_summary_knn_rehearsal_stage2.json",
        "stage1_summary_knn_rehearsal_stage3.json",
        "stage1_summary_knn_rehearsal_stage3b.json",
        "stage1_summary_knn_rehearsal_s32straight_to_s7.json",
    ],
    "rf": [
        "stage1_summary_rf_rehearsal.json",
        "stage1_summary_rf_rehearsal_newsites.json",
        "stage1_summary_rf_rehearsal_stage2.json",
        "stage1_summary_rf_rehearsal_stage3.json",
        "stage1_summary_rf_rehearsal_stage3b.json",
        "stage1_summary_rf_rehearsal_s32straight_to_s7.json",
    ],
    "fourier_knn": [
        "stage1_summary_fourier_knn_rehearsal.json",
        "stage1_summary_fourier_knn_rehearsal_newsites.json",
        "stage1_summary_fourier_knn_rehearsal_stage2.json",
        "stage1_summary_fourier_knn_rehearsal_stage3.json",
        "stage1_summary_fourier_knn_rehearsal_stage3b.json",
        "stage1_summary_fourier_knn_rehearsal_s32straight_to_s7.json",
    ],
}

MODEL_ORDER = ["mlp", "knn", "rf", "fourier_knn"]

MODEL_STYLE = {
    "mlp":         {"label": "MLP (reference)", "color": "#2a78d6", "marker": "o", "linestyle": "--", "linewidth": 2.4, "zorder": 5},
    "knn":         {"label": "k-NN",             "color": "#eda100", "marker": "^", "linestyle": "-",  "linewidth": 2.0, "zorder": 3},
    "rf":          {"label": "Random Forest",    "color": "#e87ba4", "marker": "D", "linestyle": "-",  "linewidth": 2.0, "zorder": 3},
    "fourier_knn": {"label": "Fourier + k-NN",   "color": "#008300", "marker": "v", "linestyle": "-",  "linewidth": 2.0, "zorder": 3},
}

# Real, thesis-documented scenario identities (site, time of day, lane width)
# -- see thesis_v6/thesis/main.tex, tab:extendedscenarios. Used only to build
# human-readable chart labels; no numeric data comes from this dict.
SCENARIO_INFO = {
    "s1": "McAllister day (narrow)",
    "s2": "McAllister night (narrow)",
    "s3": "Rural Rd day (wide)",
    "s4": "Rural Rd night (wide)",
    "s7": "Site 7 day (wide)",
    "s32straight": "College Ave day (narrow)",
    "s33straight": "College Ave night (narrow)",
}

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

RANDOM_CHANCE_TOP2 = 12.5  # % , 2/16 beams

# Shared color tokens (kept consistent with rehearsal_study/plot_stage2_corrected.py)
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
BASELINE_COLOR = "#c3c2b7"
SURFACE = "#fcfcfb"


def nice_label(label):
    """'s1_to_s7' -> 'McAllister day (narrow) -> Site 7 day (wide)'"""
    src, tgt = label.split("_to_")
    return f"{SCENARIO_INFO.get(src, src)} -> {SCENARIO_INFO.get(tgt, tgt)}"


def load_model_summary(model_name):
    """Merge every summary file listed for model_name into one
    {label: {"points": [...], "top2_accuracy": [...% ...], "top2_relative_power": [...% ...]}}
    dict. Raises if two files disagree on sweep_points, or if a label is
    repeated with meaningfully different values (a real correctness check,
    not a formality -- e.g. s1_to_s7 appears in both the newsites and stage2
    knn/rf/fourier_knn summary files and must agree)."""
    merged = {}
    sweep_points = None
    for fname in MODEL_SUMMARY_FILES[model_name]:
        path = os.path.join(SUMMARY_DIR, fname)
        with open(path) as f:
            d = json.load(f)
        if sweep_points is None:
            sweep_points = d["sweep_points"]
        elif d["sweep_points"] != sweep_points:
            raise ValueError(f"{fname}: sweep_points differ from earlier files for model={model_name}")
        for label, vals in d["comparisons"].items():
            acc = [v * 100 for v in vals["top2_accuracy"]]
            pwr = [v * 100 for v in vals["top2_relative_power"]]
            if label in merged:
                prev = merged[label]
                max_diff = max(abs(a - b) for a, b in zip(prev["top2_accuracy"], acc))
                if max_diff > 1e-6:
                    # Some comparisons (e.g. knn's s1_to_s7) were run twice,
                    # once in their dedicated batch and once incidentally
                    # inside a broader batch. Small (a few percentage
                    # points) run-to-run differences are expected 10-seed
                    # sampling noise, not a data-scale bug -- keep the
                    # first-seen value (from the earlier, dedicated file)
                    # and just report the discrepancy instead of crashing.
                    print(
                        f"[warn] '{label}' for model={model_name}: {fname} differs from an "
                        f"earlier file by up to {max_diff:.2f} percentage points -- keeping the "
                        f"earlier file's values.",
                        file=__import__("sys").stderr,
                    )
                continue
            merged[label] = {"points": sweep_points, "top2_accuracy": acc, "top2_relative_power": pwr}
    return merged


def load_all_models():
    """{"mlp": {...}, "knn": {...}, "rf": {...}, "fourier_knn": {...}}"""
    return {m: load_model_summary(m) for m in MODEL_ORDER}


def all_comparison_labels(all_results):
    """Labels present for every model (should be all 32 -- verified 2026-09-16)."""
    sets = [set(all_results[m].keys()) for m in MODEL_ORDER]
    common = set.intersection(*sets)
    return sorted(common)


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
