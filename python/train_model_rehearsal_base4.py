"""
MLP-only true-rehearsal batch for the four original scenario comparisons.

At N=0, each seed trains on the full source training pool and evaluates on
one fixed held-out target test partition. At every N>0 sweep point, the model
continues from that source-trained checkpoint using a rehearsal dataset made
from a capped random source replay buffer (500 samples, drawn once per seed)
plus N target-training samples. The cap matches the established MLP rehearsal
runs in this repository and keeps the long study computationally bounded;
zero-shot pretraining remains on the full source pool.

This batch intentionally runs MLP only. It writes its own summary JSON so it
can run alongside other rehearsal batches without a read/modify/write race.
Run from digital_twin_extended/:
    python python/train_model_rehearsal_base4.py
"""
import json
import os
import traceback

import torch

from train_model_cross_scenario import load_global_normalization
from train_model_rehearsal_extended import (
    COMPARISONS as UNUSED_COMPARISONS,
    SWEEP_POINTS,
    N_SEEDS,
    run_comparison,
)

RESULT_DIR = "result"
COMPARISONS = [
    ("scenario1", "scenario2", "s1_to_s2"),
    ("scenario1", "scenario3", "s1_to_s3"),
    ("scenario2", "scenario4", "s2_to_s4"),
    ("scenario3", "scenario4", "s3_to_s4"),
]
SUMMARY_PATH = os.path.join(RESULT_DIR, "stage1_summary_rehearsal_base4.json")


if __name__ == "__main__":
    torch.manual_seed(2022)
    os.makedirs(RESULT_DIR, exist_ok=True)
    max_xy, max_dist = load_global_normalization()
    print(f"Using global (7-scenario) normalization: max_xy={max_xy:.4f}, max_dist={max_dist:.4f}", flush=True)
    print(f"Running {len(COMPARISONS)} MLP rehearsal comparisons, {N_SEEDS} seeds each.", flush=True)
    print("Source replay cap: 500 samples per seed; zero-shot uses the full source pool.", flush=True)

    summary = {}
    for source, target, label in COMPARISONS:
        try:
            acc, pwr = run_comparison(source, target, label, max_xy, max_dist)
        except Exception:
            print(f"!! Comparison {label} FAILED entirely -- skipping, continuing to next comparison.", flush=True)
            traceback.print_exc()
            continue

        if acc is None:
            continue

        summary[label] = {
            "top2_accuracy": acc[1].mean(axis=-1).tolist(),
            "top2_relative_power": pwr[1].mean(axis=-1).tolist(),
        }
        with open(SUMMARY_PATH, "w") as summary_file:
            json.dump({"sweep_points": [0] + SWEEP_POINTS, "comparisons": summary}, summary_file, indent=2)

    print("\n=== Base-4 MLP rehearsal summary: mean top-2 accuracy ===", flush=True)
    for label, values in summary.items():
        print(f"{label:12s} 0={values['top2_accuracy'][0] * 100:6.2f}% "
              f"100={values['top2_accuracy'][20] * 100:6.2f}% "
              f"200={values['top2_accuracy'][-1] * 100:6.2f}%", flush=True)
    print(f"\nCompleted {len(summary)}/{len(COMPARISONS)} comparisons successfully.", flush=True)
    print("Base-4 MLP rehearsal batch complete.", flush=True)
