"""
Filters Scenario32/33 (College Ave) down to the straight segment of the
road only, excluding the bend visible in the existing scatter plots
(data/scenario32_ue_scatter.png, data/scenario33_ue_scatter.png).

Why: per-y-bin analysis of the raw relative position showed the x-spread
(std) stays tight and stable (~1.5-2.4 m) for y < 40 m, then roughly
doubles/triples for y >= 40 m -- the clear geometric signature of the road
curving away from its dominant heading. y=40 (in raw, unnormalized relative
meters) is used as the cutoff for both scenarios based on this.

This is a SEPARATE, additive preprocessing step -- it does not touch or
overwrite the original scenario32_*/scenario33_* files (used for the
"not cleaned" / bending-included results already computed), and produces
a second, parallel set of files under the "scenario32straight" /
"scenario33straight" pseudo-scenario names, loadable via the same
paths_for()/DataFeed convention used everywhere else in this project:
  data/scenario32straight_ue_relative_pos.mat
  data/scenario32straight_real_beam_pwr.mat
  data/scenario32straight_ue_gps_pos.mat
  (and the same three for scenario33straight)

Both the original (bend-included) and new (straight-only) scatter plots
are kept side by side, for the thesis, to visually show what was removed.

Run from the repository root:
    python python/preprocess_college_ave_straight.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.io import loadmat, savemat

DATA_DIR = "data"
Y_CUTOFF = 40.0  # meters, raw relative y -- see docstring for how this was chosen


def load_raw(scenario):
    pos = loadmat(os.path.join(DATA_DIR, f"{scenario}_ue_relative_pos.mat"))["ue_relative_pos"]
    pwr = loadmat(os.path.join(DATA_DIR, f"{scenario}_real_beam_pwr.mat"))["real_beam_pwr"]
    gps = loadmat(os.path.join(DATA_DIR, f"{scenario}_ue_gps_pos.mat"))["ue_gps_pos"]
    return pos, pwr, gps


def save_scatter(pos, title, fig_path):
    plt.figure()
    plt.scatter(pos[:, 0], pos[:, 1], s=0.5)
    plt.scatter(0, 0, c="red", marker="x", label="BS")
    plt.axis("equal")
    plt.xlabel("X-coordinates (meter)")
    plt.ylabel("Y-coordinates (meter)")
    x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
    y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                      linewidth=1, edgecolor="r", facecolor="none")
    plt.gca().add_patch(rect)
    plt.legend()
    plt.title(title)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig_path}")


def process(scenario_label, display_name):
    pos, pwr, gps = load_raw(scenario_label)
    n_total = pos.shape[0]

    # Original (bend-included) scatter -- redrawn here only for a matching
    # side-by-side pair; the pipeline's own original plot at
    # data/{scenario}_ue_scatter.png is untouched and still the canonical one.
    save_scatter(pos, f"{display_name}: UE positions relative to BS (full, bend included)",
                 os.path.join(DATA_DIR, f"{scenario_label}_full_ue_scatter.png"))

    straight_mask = pos[:, 1] < Y_CUTOFF
    n_straight = straight_mask.sum()
    n_bend = n_total - n_straight
    print(f"{scenario_label}: {n_total} total, {n_straight} straight (kept, y<{Y_CUTOFF}), "
          f"{n_bend} in the bend (dropped, y>={Y_CUTOFF})")

    pos_s, pwr_s, gps_s = pos[straight_mask], pwr[straight_mask], gps[straight_mask]

    out_label = f"{scenario_label}straight"
    savemat(os.path.join(DATA_DIR, f"{out_label}_ue_relative_pos.mat"), {"ue_relative_pos": pos_s})
    savemat(os.path.join(DATA_DIR, f"{out_label}_real_beam_pwr.mat"), {"real_beam_pwr": pwr_s})
    savemat(os.path.join(DATA_DIR, f"{out_label}_ue_gps_pos.mat"), {"ue_gps_pos": gps_s})
    print(f"Saved straight-only .mat files under data/{out_label}_*")

    save_scatter(pos_s, f"{display_name}: UE positions relative to BS (straight segment only, y<{Y_CUTOFF}m)",
                 os.path.join(DATA_DIR, f"{out_label}_ue_scatter.png"))

    return n_total, n_straight, n_bend


if __name__ == "__main__":
    summary = {}
    for scen, name in [("scenario32", "Scenario32 (College Ave, day)"),
                        ("scenario33", "Scenario33 (College Ave, night)")]:
        n_total, n_straight, n_bend = process(scen, name)
        summary[scen] = {"n_total": int(n_total), "n_straight": int(n_straight), "n_bend": int(n_bend)}

    print("\n=== Summary ===")
    for scen, s in summary.items():
        print(f"{scen}: kept {s['n_straight']}/{s['n_total']} "
              f"({100*s['n_straight']/s['n_total']:.1f}%), dropped {s['n_bend']} in the bend")
