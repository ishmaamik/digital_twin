"""Run the missing Stage-2 rehearsal comparison: 32straight -> 7.

Scope is intentionally limited to knn, rf, and fourier_knn. The imported
runner uses full-source rehearsal, equal aggregate RF sample weights, and
source-sized target oversampling for KNN/Fourier-KNN. Existing result files
are not overwritten except for the six missing model-specific outputs created
by this run.

Run from digital_twin_extended/:
    python rehearsal_study/run_missing_s32straight_to_s7.py --model knn
    python rehearsal_study/run_missing_s32straight_to_s7.py --model rf
    python rehearsal_study/run_missing_s32straight_to_s7.py --model fourier_knn
"""
import argparse
import json
import os

from train_baselines_cross_scenario_rehearsal import (
    SWEEP_POINTS,
    load_global_normalization,
    run_comparison,
)

RESULT_DIR = "result"
SOURCE = "scenario32straight"
TARGET = "scenario7"
LABEL = "s32straight_to_s7"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["knn", "rf", "fourier_knn"], required=True)
    args = parser.parse_args()

    max_xy, max_dist = load_global_normalization()
    print(f"Using normalization: max_xy={max_xy:.4f}, max_dist={max_dist:.4f}", flush=True)
    print(f"Running missing rehearsal comparison {SOURCE} -> {TARGET} with model={args.model}", flush=True)

    acc, pwr = run_comparison(args.model, SOURCE, TARGET, LABEL, max_xy, max_dist)
    summary = {
        "model": args.model,
        "protocol": "rehearsal",
        "sweep_points": [0] + SWEEP_POINTS,
        "comparisons": {
            LABEL: {
                "top2_accuracy": acc[1].mean(axis=-1).tolist(),
                "top2_relative_power": pwr[1].mean(axis=-1).tolist(),
            }
        },
    }
    summary_path = os.path.join(RESULT_DIR, f"stage1_summary_{args.model}_rehearsal_s32straight_to_s7.json")
    with open(summary_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(f"Saved {summary_path}", flush=True)
    print(f"Rehearsal baseline comparison complete for model={args.model}.", flush=True)


if __name__ == "__main__":
    main()
