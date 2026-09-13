"""
author:Shuaifeng
data:10/8/2022

Modified for the post-deployment environmental-shift study (Phase 0, Step 1):
generalized from a single hardcoded scenario to loop over all four DeepSense6G
scenarios used in Stage 1 (McAllister Ave day/night, Rural Road day/night),
now located under datasets/ instead of at the repo root.
"""
# %%
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import utm
from scipy.io import savemat
from tqdm import tqdm

N_BEAMS = 64
# Boresight beam index of the 16-element / 64-beam codebook. This is a property
# of the shared hardware, not of any one street, so it is not re-tuned per scenario.
CENTER_BEAM = 28

# The four scenarios now live under datasets/, not at the repo root.
SCENARIO_FOLDERS = [
    "datasets/Scenario1",
    "datasets/Scenario2",
    "datasets/Scenario3",
    "datasets/Scenario4",
]

OUTPUT_DIR = "data"


def xy_from_latlong(lat_long):
    """Assumes lat and long along row. Returns same row vec/matrix on
    cartesian coords (UTM easting, northing in meters)."""
    # utm.from_latlon() returns: (EASTING, NORTHING, ZONE_NUMBER, ZONE_LETTER)
    x, y, *_ = utm.from_latlon(lat_long[:, 0], lat_long[:, 1])
    return np.stack((x, y), axis=1)


def preprocess_scenario(scenario_folder):
    """Run position/beam preprocessing for one scenario folder.

    Saves per-scenario .mat outputs into OUTPUT_DIR, prefixed with the
    scenario's own name so scenarios never overwrite each other, and so the
    original unprefixed data/*.mat files already used by the Stage-0
    reproduction scripts are left untouched.
    """
    scenario_name = os.path.basename(scenario_folder.rstrip("/\\"))
    print(f"\n=== Processing {scenario_name} ({scenario_folder}) ===")

    try:
        csv_file = [f for f in os.listdir(scenario_folder) if f.endswith("csv")][0]
    except (FileNotFoundError, IndexError):
        raise Exception(f"No csv file inside {scenario_folder}.")
    csv_path = os.path.join(scenario_folder, csv_file)

    dataframe = pd.read_csv(csv_path)
    n_samples = dataframe.index.stop
    print(f"Number of real data points: {n_samples}")

    # Load beam power sweeps
    pwr_rel_paths = dataframe["unit1_pwr_60ghz"].values
    pwrs_array = np.zeros((n_samples, N_BEAMS))
    for sample_idx in tqdm(range(n_samples), desc="Loading beam power"):
        pwr_abs_path = os.path.join(scenario_folder, pwr_rel_paths[sample_idx])
        pwrs_array[sample_idx] = np.loadtxt(pwr_abs_path)

    best_beams = np.argmax(pwrs_array, 1) + 1  # 1-indexed, out of 64

    # Load BS position (fixed, single point) and UE positions
    pos_bs = np.loadtxt(os.path.join(scenario_folder, dataframe["unit1_loc"].values[0]))
    pos_bs = np.expand_dims(pos_bs, 0)

    pos_rel_paths = dataframe["unit2_loc"].values
    pos_ue_array = np.zeros((n_samples, 2))  # 2 = Latitude and Longitude
    for sample_idx in range(n_samples):
        pos_abs_path = os.path.join(scenario_folder, pos_rel_paths[sample_idx])
        pos_ue_array[sample_idx] = np.loadtxt(pos_abs_path)

    pos_ue_cart = xy_from_latlong(pos_ue_array)
    pos_bs_cart = xy_from_latlong(pos_bs)
    pos_diff = pos_ue_cart - pos_bs_cart

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prefix = scenario_name.lower()
    savemat(os.path.join(OUTPUT_DIR, f"{prefix}_ue_relative_pos.mat"), {"ue_relative_pos": pos_diff})
    savemat(os.path.join(OUTPUT_DIR, f"{prefix}_real_beam_pwr.mat"), {"real_beam_pwr": pwrs_array})
    savemat(os.path.join(OUTPUT_DIR, f"{prefix}_ue_gps_pos.mat"), {"ue_gps_pos": pos_ue_array})
    print(f"Saved position and beam power .mat files for {scenario_name} to {OUTPUT_DIR}/")

    # Sanity-check scatter plot: BS at origin, UE cloud, single bounding box.
    # The original script's two-rectangle split (at a hardcoded x < 18 threshold)
    # was specific to McAllister Ave's two street segments and is not reused here,
    # since it would not generalize correctly to Rural Road's different layout.
    plt.figure()
    plt.scatter(pos_diff[:, 0], pos_diff[:, 1], s=0.5)
    plt.scatter(0, 0, c="red", marker="x", label="BS")
    plt.axis("equal")
    plt.xlabel("X-coordinates (meter)")
    plt.ylabel("Y-coordinates (meter)")
    x_min, x_max = pos_diff[:, 0].min(), pos_diff[:, 0].max()
    y_min, y_max = pos_diff[:, 1].min(), pos_diff[:, 1].max()
    rect = Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=1, edgecolor="r", facecolor="none",
    )
    plt.gca().add_patch(rect)
    plt.legend()
    plt.title(f"{scenario_name}: UE positions relative to BS")
    fig_path = os.path.join(OUTPUT_DIR, f"{prefix}_ue_scatter.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot to {fig_path}")

    # BS orientation via the center-beam least-squares trick: k = sum(x_i*y_i) / sum(x_i^2).
    # Unchanged from the original Scenario-1-only script; CENTER_BEAM is a hardware
    # constant, so this logic is already scenario-general.
    center_ue = np.where(best_beams == CENTER_BEAM)[0]
    bs_orientation = None
    if len(center_ue) == 0:
        print(f"WARNING: no samples in {scenario_name} have best beam == {CENTER_BEAM}; "
              f"BS orientation cannot be estimated for this scenario.")
    else:
        center_ue_pos = pos_diff[center_ue, :]
        denom = np.sum(center_ue_pos[:, 0] * center_ue_pos[:, 0])
        if denom == 0:
            print(f"WARNING: degenerate center-beam geometry in {scenario_name}; "
                  f"BS orientation cannot be estimated.")
        else:
            k = np.sum(center_ue_pos[:, 1] * center_ue_pos[:, 0]) / denom
            bs_orientation = float(np.arctan(k) / np.pi * 180)
            print(f"{scenario_name} BS orientation: {bs_orientation:.2f} degrees "
                  f"(from {len(center_ue)} center-beam samples)")

    return {
        "scenario": scenario_name,
        "n_samples": int(n_samples),
        "n_center_beam_samples": int(len(center_ue)),
        "bs_orientation_deg": bs_orientation,
        "pos_diff": pos_diff,  # kept in-memory only; stripped before saving JSON below
    }


if __name__ == "__main__":
    results = [preprocess_scenario(folder) for folder in SCENARIO_FOLDERS]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # JSON-safe summary (drop the raw position arrays before dumping).
    summaries = [{k: v for k, v in r.items() if k != "pos_diff"} for r in results]
    summary_path = os.path.join(OUTPUT_DIR, "scenario_preprocessing_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n=== Summary across all scenarios ===")
    for s in summaries:
        print(s)
    print(f"\nFull summary saved to {summary_path}")

    # Global normalization constants, pooled across all four scenarios, used by
    # data_feed.py's create_samples()/DataFeed for the Stage-1 multi-scenario
    # study. Kept separate from the original 30 / sqrt(24**2+28**2) constants,
    # which remain the defaults used for the unmodified Stage-0 reproduction.
    all_pos_diff = np.concatenate([r["pos_diff"] for r in results], axis=0)
    global_max_xy = float(np.max(np.abs(all_pos_diff)))
    distances = np.sqrt(all_pos_diff[:, 0] ** 2 + all_pos_diff[:, 1] ** 2)
    global_max_dist = float(np.max(distances))

    global_norm = {
        "max_xy": global_max_xy,
        "max_dist": global_max_dist,
        "pooled_from_scenarios": [r["scenario"] for r in results],
        "total_samples_pooled": int(all_pos_diff.shape[0]),
    }
    norm_path = os.path.join(OUTPUT_DIR, "global_normalization.json")
    with open(norm_path, "w") as f:
        json.dump(global_norm, f, indent=2)

    print("\n=== Global normalization constants (pooled across all scenarios) ===")
    print(global_norm)
    print(f"Saved to {norm_path}")
    print("Preprocessing complete for all scenarios.")
