"""Run all remaining baseline rehearsal comparisons with a 500-source cap.

This is KNN, RF, and Fourier-KNN only. It covers the 32 directed pairs in
 the completed cross-scenario matrix, including the four raw-bending pairs
where applicable. Results are written only to result/Other_models_500-cap,
so existing result files and the earlier 500-cap folder are untouched.

The imported runner fixes a 500-sample source replay buffer per seed, uses
 equal aggregate source/target weighting for RF, and target oversampling for
 KNN/Fourier-KNN. Run from digital_twin_extended/ with one model per process:
    python rehearsal_study/run_all_other_models_cap500.py --model knn
    python rehearsal_study/run_all_other_models_cap500.py --model rf
    python rehearsal_study/run_all_other_models_cap500.py --model fourier_knn
"""
import argparse
import json
import os

import run_straight_cap500_rehearsal as cap500

RESULT_DIR = os.path.join("result", "Other_models_500-cap")
COMPARISONS = [
    ("scenario1", "scenario2", "s1_to_s2"),
    ("scenario1", "scenario3", "s1_to_s3"),
    ("scenario1", "scenario7", "s1_to_s7"),
    ("scenario1", "scenario32straight", "s1_to_s32straight"),
    ("scenario1", "scenario33straight", "s1_to_s33straight"),
    ("scenario2", "scenario1", "s2_to_s1"),
    ("scenario2", "scenario4", "s2_to_s4"),
    ("scenario2", "scenario7", "s2_to_s7"),
    ("scenario2", "scenario32straight", "s2_to_s32straight"),
    ("scenario2", "scenario33straight", "s2_to_s33straight"),
    ("scenario3", "scenario1", "s3_to_s1"),
    ("scenario3", "scenario4", "s3_to_s4"),
    ("scenario3", "scenario7", "s3_to_s7"),
    ("scenario3", "scenario32straight", "s3_to_s32straight"),
    ("scenario4", "scenario2", "s4_to_s2"),
    ("scenario4", "scenario3", "s4_to_s3"),
    ("scenario4", "scenario7", "s4_to_s7"),
    ("scenario4", "scenario33straight", "s4_to_s33straight"),
    ("scenario7", "scenario1", "s7_to_s1"),
    ("scenario7", "scenario2", "s7_to_s2"),
    ("scenario7", "scenario3", "s7_to_s3"),
    ("scenario7", "scenario4", "s7_to_s4"),
    ("scenario7", "scenario32straight", "s7_to_s32straight"),
    ("scenario32straight", "scenario1", "s32straight_to_s1"),
    ("scenario32straight", "scenario2", "s32straight_to_s2"),
    ("scenario32straight", "scenario3", "s32straight_to_s3"),
    ("scenario32straight", "scenario7", "s32straight_to_s7"),
    ("scenario32straight", "scenario33straight", "s32straight_to_s33straight"),
    ("scenario33straight", "scenario1", "s33straight_to_s1"),
    ("scenario33straight", "scenario2", "s33straight_to_s2"),
    ("scenario33straight", "scenario4", "s33straight_to_s4"),
    ("scenario33straight", "scenario32straight", "s33straight_to_s32straight"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["knn", "rf", "fourier_knn"], required=True)
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    cap500.RESULT_DIR = RESULT_DIR
    cap500.COMPARISONS = COMPARISONS
    max_xy, max_dist = cap500.load_global_normalization()
    print(f"Using source cap={cap500.SOURCE_CAP}; model={args.model}", flush=True)
    print(f"Writing results to {RESULT_DIR}; comparisons={len(COMPARISONS)}", flush=True)

    summary = {}
    for source, target, label in COMPARISONS:
        acc, pwr = cap500.run_comparison(args.model, source, target, label, max_xy, max_dist)
        summary[label] = {
            "top2_accuracy": acc[1].mean(axis=-1).tolist(),
            "top2_relative_power": pwr[1].mean(axis=-1).tolist(),
        }
        with open(os.path.join(RESULT_DIR, f"stage1_summary_{args.model}_rehearsal.json"), "w") as summary_file:
            json.dump({
                "model": args.model,
                "protocol": "rehearsal_source_cap_500",
                "sweep_points": [0] + cap500.SWEEP_POINTS,
                "comparisons": summary,
            }, summary_file, indent=2)

    print(f"Rehearsal cap-500 all-comparisons complete for model={args.model}.", flush=True)


if __name__ == "__main__":
    main()
