"""
Non-neural baselines for the Stage 1 cross-scenario comparison: k-NN and
Random Forest. Added alongside the neural architectures (see model.py /
train_model_cross_scenario.py) to give the resource-efficiency comparison a
near-zero-training-cost reference point.

These don't have a natural pretrain-then-fine-tune mechanic (no gradient-based
warm start), so each sweep point is a *fresh* fit instead of a continued one:
  - 0 real target samples: fit on the full source training set, evaluate on
    the held-out target test set. Directly comparable to the neural zero-shot
    step, since it uses the exact same source/target split.
  - N real target samples: fit from scratch on those N target-only samples,
    evaluate on the same held-out target test set. This isolates "how many
    real target samples does a cheap baseline need", the same question
    train_model_cross_scenario.py asks of the neural models.

Note the per-seed train/test partitions here are drawn from the same 80/20
split methodology as the neural runs but are not seed-for-seed identical to
them (the neural scripts' RNG state also gets consumed by model-weight
initialization, so even the 5 neural architectures don't share identical
per-seed splits with each other). All runs still use N_SEEDS independent
random splits, so mean-over-seeds comparisons remain fair.

Must be run from the repository root, e.g.:
    python model_comparison_study/train_baselines_cross_scenario.py --model knn
    python model_comparison_study/train_baselines_cross_scenario.py --model rf
    python model_comparison_study/train_baselines_cross_scenario.py --model fourier_knn
    python model_comparison_study/train_baselines_cross_scenario.py --model fourier_rf

fourier_knn / fourier_rf feed the classifier a NeRF/SIREN-style sinusoidal
(x, y) encoding instead of raw coordinates -- a fast, cheap way to test
whether spatial encoding helps, without paying for a full neural training
loop, and without the neural pipeline's own source-pretrain/target-finetune
weakness confounding the result (see knn/rf's results vs. the neural models).
"""
import argparse
import json
import os

import numpy as np
from scipy.io import savemat
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from data_feed import create_samples
from train_model_cross_scenario import (
    COMPARISONS,
    RESULT_DIR,
    SWEEP_POINTS,
    N_SEEDS,
    load_global_normalization,
    paths_for,
)

NUM_CLASSES = 16
FOURIER_NUM_FREQUENCIES = 6

BASELINE_MODELS = {
    "knn": lambda n_train: KNeighborsClassifier(n_neighbors=min(5, max(1, n_train))),
    "rf": lambda n_train: RandomForestClassifier(n_estimators=200, random_state=0),
}

# "fourier_knn" / "fourier_rf": same classifier, fed a NeRF/SIREN-style
# sinusoidal (x, y) encoding instead of raw coordinates. A fast (~1-5 min)
# proxy for "does spatial encoding help", without paying for a full neural
# training loop -- and one that isolates the encoding's effect from the
# neural pipeline's own source-pretrain/target-finetune weakness (see
# knn/rf's results vs. the neural models).
FOURIER_PREFIX = "fourier_"


def fourier_encode(pos):
    xy = pos[:, :2]
    freq_bands = 2.0 ** np.arange(FOURIER_NUM_FREQUENCIES)
    terms = [pos]
    for freq in freq_bands:
        terms.append(np.sin(xy * freq * np.pi))
        terms.append(np.cos(xy * freq * np.pi))
    return np.concatenate(terms, axis=-1)


def load_split(pos_path, pwr_path, rand_state, mode, max_xy, max_dist, num_data_point=None, fourier=False):
    pos, labels, pwr = create_samples(
        pos_path, pwr_path, rand_state, mode, train_split=0.8,
        num_data_point=num_data_point, portion=1., max_xy=max_xy, max_dist=max_dist,
    )
    if fourier:
        pos = fourier_encode(pos)
    return pos, labels, pwr


def topk_metrics(clf, X, labels, pwr, k_list=(1, 2, 3, 5)):
    proba = clf.predict_proba(X)
    full_proba = np.zeros((X.shape[0], NUM_CLASSES))
    full_proba[:, clf.classes_] = proba
    order = np.argsort(-full_proba, axis=1)

    acc = []
    pwr_ratio = []
    max_pwr = pwr[np.arange(len(labels)), labels]
    for k in k_list:
        topk = order[:, :k]
        correct = np.any(topk == labels[:, None], axis=1)
        acc.append(correct.mean())

        topk_pwr = np.take_along_axis(pwr, topk, axis=1).max(axis=1)
        pwr_ratio.append((topk_pwr / max_pwr).mean())

    return np.array(acc), np.array(pwr_ratio)


def run_comparison(model_name, source, target, label, max_xy, max_dist):
    print(f"\n{'=' * 70}\nBaseline [{model_name}] comparison: {source} -> {target}  ({label})\n{'=' * 70}", flush=True)
    fourier = model_name.startswith(FOURIER_PREFIX)
    base_model = model_name[len(FOURIER_PREFIX):] if fourier else model_name
    source_pos, source_pwr = paths_for(source)
    target_pos, target_pwr = paths_for(target)

    all_acc = []
    all_pwr = []

    for seed_idx in range(N_SEEDS):
        rand_state = 1000 + seed_idx  # deterministic, independent of the neural scripts' torch RNG

        target_test_pos, target_test_labels, target_test_pwr = load_split(
            target_pos, target_pwr, rand_state, "test", max_xy, max_dist, fourier=fourier,
        )

        acc_this_seed = []
        pwr_this_seed = []

        # 0 real target samples: fit on the full source train set.
        source_train_pos, source_train_labels, _ = load_split(
            source_pos, source_pwr, rand_state, "train", max_xy, max_dist, fourier=fourier,
        )
        clf = BASELINE_MODELS[base_model](len(source_train_labels))
        clf.fit(source_train_pos, source_train_labels)
        acc, pwr_ratio = topk_metrics(clf, target_test_pos, target_test_labels, target_test_pwr)
        acc_this_seed.append(acc)
        pwr_this_seed.append(pwr_ratio)

        # Increasing real target samples: fresh fit each time, target-only.
        for num_data_point in SWEEP_POINTS:
            target_train_pos, target_train_labels, _ = load_split(
                target_pos, target_pwr, rand_state, "train", max_xy, max_dist,
                num_data_point=num_data_point, fourier=fourier,
            )
            clf = BASELINE_MODELS[base_model](len(target_train_labels))
            clf.fit(target_train_pos, target_train_labels)
            acc, pwr_ratio = topk_metrics(clf, target_test_pos, target_test_labels, target_test_pwr)
            acc_this_seed.append(acc)
            pwr_this_seed.append(pwr_ratio)

        all_acc.append(np.stack(acc_this_seed, -1))  # (4 metrics, 1+len(SWEEP_POINTS))
        all_pwr.append(np.stack(pwr_this_seed, -1))

    all_acc = np.stack(all_acc, -1)  # (4 metrics, points, seeds)
    all_pwr = np.stack(all_pwr, -1)

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_acc_path = os.path.join(RESULT_DIR, f"stage1_{label}_{model_name}_acc.mat")
    out_pwr_path = os.path.join(RESULT_DIR, f"stage1_{label}_{model_name}_pwr.mat")
    savemat(out_acc_path, {"acc": all_acc, "sweep_points": [0] + SWEEP_POINTS})
    savemat(out_pwr_path, {"pwr": all_pwr, "sweep_points": [0] + SWEEP_POINTS})
    print(f"Saved {out_acc_path} and {out_pwr_path}", flush=True)

    return all_acc, all_pwr


def main():
    all_model_choices = list(BASELINE_MODELS.keys()) + [FOURIER_PREFIX + k for k in BASELINE_MODELS.keys()]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=all_model_choices, required=True)
    args = parser.parse_args()

    max_xy, max_dist = load_global_normalization()
    print(f"Using global normalization: max_xy={max_xy:.4f}, max_dist={max_dist:.4f}", flush=True)
    print(f"Running Stage 1 baseline with model={args.model}", flush=True)

    summary = {}
    for source, target, label in COMPARISONS:
        acc, pwr = run_comparison(args.model, source, target, label, max_xy, max_dist)
        top2_acc = acc[1].mean(axis=-1)
        top2_pwr = pwr[1].mean(axis=-1)
        summary[label] = {
            "top2_accuracy": top2_acc.tolist(),
            "top2_relative_power": top2_pwr.tolist(),
        }

    points = [0] + SWEEP_POINTS
    print(f"\n=== Baseline [{args.model}] Stage 1 summary: mean top-2 accuracy per comparison ===", flush=True)
    header = "real_samples".rjust(12) + "".join(f"{lbl:>28}" for lbl in summary)
    print(header, flush=True)
    for i, p in enumerate(points):
        row = str(p).rjust(12) + "".join(f"{summary[lbl]['top2_accuracy'][i] * 100:>27.2f}%" for lbl in summary)
        print(row, flush=True)

    with open(os.path.join(RESULT_DIR, f"stage1_summary_{args.model}.json"), "w") as f:
        json.dump({"model": args.model, "sweep_points": points, "comparisons": summary}, f, indent=2)

    print("\nBaseline comparison complete.", flush=True)


if __name__ == "__main__":
    main()
