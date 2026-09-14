"""
Rehearsal-protocol Stage-2 study: k-NN, Random Forest, and their
Fourier-encoded variants.

Replaces the earlier target-only-fit protocol (see the sibling
model_comparison_study/ code on another branch) with REHEARSAL: at every
sweep point N > 0, each classifier is fit on the FULL source-scenario
dataset UNION the N real target-scenario samples for that point -- never
on target-only data, and never (past N=0) on source-only data. This was
necessary because, under the old target-only protocol, two comparisons
sharing the same target scenario produced byte-identical results at every
N > 0 (confirmed directly by diffing stage1_time_of_day_at_ruralroad_*.mat
against stage1_site_during_night_*.mat, both targeting Scenario 4), since
fitting on target-only data threw the source scenario away entirely once
N > 0, leaving it no way to influence anything past the zero-shot point.

Protocol, per seed:
  1. Zero-shot (N=0): fit on the FULL source-scenario dataset (100% of its
     samples -- no train/test split on the source, since it is training
     input only and is never evaluated). Evaluate on the target scenario's
     held-out test partition (80/20 split, mode='test'); this partition is
     fixed once per seed and reused for every sweep point below.
  2. For each N in SWEEP_POINTS: draw N samples from the target scenario's
     80% train pool. Fit a FRESH classifier (none of these models support
     a gradient-style warm start) on the UNION of the full source dataset
     and these N target samples, then evaluate on the same fixed target
     test partition.

Source/target imbalance -- EQUAL AGGREGATE WEIGHT scheme, applied
consistently across both models kept here:
  - Random Forest: sample_weight gives every source row weight 1/n_source
    and every target row weight 1/N, so the two pools sum to equal total
    weight (1.0 each) in the fit regardless of how small N is.
  - k-NN / Fourier-k-NN: KNeighborsClassifier.fit() has no sample_weight
    parameter, so the same 50/50 principle is instead applied by
    OVERSAMPLING the N target samples WITH REPLACEMENT up to n_source
    copies before combining -- i.e. the fitted set is n_source original
    source rows plus n_source resampled-with-replacement target rows, an
    explicit, documented choice (not silent unweighted rehearsal, which
    the source pool would otherwise dominate by a factor of n_source/N).

Scope: only knn, rf, and fourier_knn are kept in this study (fourier_rf
and a TinyMLP variant were dropped by request). This script is therefore
fully self-contained -- no sibling neural-model script to import shared
constants from.

Must be run from the repository root, e.g.:
    python rehearsal_study/train_baselines_rehearsal.py --model knn
    python rehearsal_study/train_baselines_rehearsal.py --model rf
    python rehearsal_study/train_baselines_rehearsal.py --model fourier_knn
"""
import argparse
import json
import os

import numpy as np
from scipy.io import savemat
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from data_feed import create_samples

DATA_DIR = "data"
RESULT_DIR = "result"

# All 8 source -> target comparisons among Scenarios 1-4 (both directions
# of each of the original 4 single-factor comparisons).
COMPARISONS = [
    ("scenario1", "scenario2", "s1_to_s2"),
    ("scenario2", "scenario1", "s2_to_s1"),
    ("scenario3", "scenario1", "s3_to_s1"),
    ("scenario4", "scenario2", "s4_to_s2"),
    ("scenario1", "scenario3", "s1_to_s3"),
    ("scenario2", "scenario4", "s2_to_s4"),
    ("scenario3", "scenario4", "s3_to_s4"),
    ("scenario4", "scenario3", "s4_to_s3"),
]

SWEEP_POINTS = list(range(5, 101, 5)) + [150, 200]
N_SEEDS = 10

NUM_CLASSES = 16
FOURIER_NUM_FREQUENCIES = 6
FOURIER_PREFIX = "fourier_"

BASELINE_MODELS = {
    "knn": lambda n_train: KNeighborsClassifier(n_neighbors=min(5, max(1, n_train))),
    "rf": lambda n_train: RandomForestClassifier(n_estimators=200, random_state=0),
}

# Only these 3 are exposed as runnable choices (fourier_rf dropped by request).
MODEL_CHOICES = ["knn", "rf", "fourier_knn"]


def load_global_normalization():
    with open(os.path.join(DATA_DIR, "global_normalization.json")) as f:
        norm = json.load(f)
    return norm["max_xy"], norm["max_dist"]


def paths_for(scenario):
    pos_path = os.path.join(DATA_DIR, f"{scenario}_ue_relative_pos.mat")
    pwr_path = os.path.join(DATA_DIR, f"{scenario}_real_beam_pwr.mat")
    return pos_path, pwr_path


def fourier_encode(pos):
    xy = pos[:, :2]
    freq_bands = 2.0 ** np.arange(FOURIER_NUM_FREQUENCIES)
    terms = [pos]
    for freq in freq_bands:
        terms.append(np.sin(xy * freq * np.pi))
        terms.append(np.cos(xy * freq * np.pi))
    return np.concatenate(terms, axis=-1)


def load_split(pos_path, pwr_path, rand_state, mode, max_xy, max_dist,
               num_data_point=None, train_split=0.8, fourier=False):
    pos, labels, pwr = create_samples(
        pos_path, pwr_path, rand_state, mode, train_split=train_split,
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


def build_rehearsal_fit_data(model_name, base_model, source_pos, source_labels, target_pos, target_labels, rng):
    """Combine the full source pool with the N target samples under the
    equal-aggregate-weight scheme, returning (X, y, sample_weight_or_None)
    ready to pass to clf.fit()."""
    n_source = source_pos.shape[0]
    n_target = target_pos.shape[0]

    if base_model == "rf":
        X = np.concatenate([source_pos, target_pos], axis=0)
        y = np.concatenate([source_labels, target_labels], axis=0)
        sample_weight = np.concatenate([
            np.full(n_source, 1.0 / n_source),
            np.full(n_target, 1.0 / n_target),
        ])
        return X, y, sample_weight

    # knn: no sample_weight support -- oversample target rows with
    # replacement up to n_source copies for equal aggregate representation.
    oversample_idx = rng.integers(0, n_target, size=n_source)
    target_pos_oversampled = target_pos[oversample_idx]
    target_labels_oversampled = target_labels[oversample_idx]
    X = np.concatenate([source_pos, target_pos_oversampled], axis=0)
    y = np.concatenate([source_labels, target_labels_oversampled], axis=0)
    return X, y, None


def run_comparison(model_name, source, target, label, max_xy, max_dist):
    print(f"\n{'=' * 70}\nRehearsal baseline [{model_name}] comparison: {source} -> {target}  ({label})\n{'=' * 70}", flush=True)
    fourier = model_name.startswith(FOURIER_PREFIX)
    base_model = model_name[len(FOURIER_PREFIX):] if fourier else model_name
    source_pos, source_pwr = paths_for(source)
    target_pos, target_pwr = paths_for(target)

    all_acc = []
    all_pwr = []

    for seed_idx in range(N_SEEDS):
        rand_state = 1000 + seed_idx
        rng = np.random.default_rng(rand_state)

        target_test_pos, target_test_labels, target_test_pwr = load_split(
            target_pos, target_pwr, rand_state, "test", max_xy, max_dist, fourier=fourier,
        )

        # Full source pool, 100% of its samples (no split -- training input only).
        source_full_pos, source_full_labels, _ = load_split(
            source_pos, source_pwr, rand_state, "train", max_xy, max_dist,
            train_split=1.0, fourier=fourier,
        )

        acc_this_seed = []
        pwr_this_seed = []

        # Step 1: zero-shot -- fit on the full source pool alone.
        clf = BASELINE_MODELS[base_model](len(source_full_labels))
        clf.fit(source_full_pos, source_full_labels)
        acc, pwr_ratio = topk_metrics(clf, target_test_pos, target_test_labels, target_test_pwr)
        acc_this_seed.append(acc)
        pwr_this_seed.append(pwr_ratio)

        # Step 2: rehearsal sweep -- fresh fit on (full source) UNION (N target).
        for num_data_point in SWEEP_POINTS:
            target_train_pos, target_train_labels, _ = load_split(
                target_pos, target_pwr, rand_state, "train", max_xy, max_dist,
                num_data_point=num_data_point, fourier=fourier,
            )
            X, y, sample_weight = build_rehearsal_fit_data(
                model_name, base_model, source_full_pos, source_full_labels,
                target_train_pos, target_train_labels, rng,
            )
            clf = BASELINE_MODELS[base_model](len(y))
            if sample_weight is not None:
                clf.fit(X, y, sample_weight=sample_weight)
            else:
                clf.fit(X, y)
            acc, pwr_ratio = topk_metrics(clf, target_test_pos, target_test_labels, target_test_pwr)
            acc_this_seed.append(acc)
            pwr_this_seed.append(pwr_ratio)

        all_acc.append(np.stack(acc_this_seed, -1))
        all_pwr.append(np.stack(pwr_this_seed, -1))

    all_acc = np.stack(all_acc, -1)
    all_pwr = np.stack(all_pwr, -1)

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_acc_path = os.path.join(RESULT_DIR, f"stage1_{label}_rehearsal_{model_name}_acc.mat")
    out_pwr_path = os.path.join(RESULT_DIR, f"stage1_{label}_rehearsal_{model_name}_pwr.mat")
    savemat(out_acc_path, {"acc": all_acc, "sweep_points": [0] + SWEEP_POINTS})
    savemat(out_pwr_path, {"pwr": all_pwr, "sweep_points": [0] + SWEEP_POINTS})
    print(f"Saved {out_acc_path} and {out_pwr_path}", flush=True)

    return all_acc, all_pwr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    args = parser.parse_args()

    max_xy, max_dist = load_global_normalization()
    print(f"Using normalization: max_xy={max_xy:.4f}, max_dist={max_dist:.4f}", flush=True)
    print(f"Running REHEARSAL Stage 2 baseline with model={args.model}", flush=True)

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
    print(f"\n=== Rehearsal baseline [{args.model}] summary: mean top-2 accuracy per comparison ===", flush=True)
    header = "real_samples".rjust(12) + "".join(f"{lbl:>14}" for lbl in summary)
    print(header, flush=True)
    for i, p in enumerate(points):
        row = str(p).rjust(12) + "".join(f"{summary[lbl]['top2_accuracy'][i] * 100:>13.2f}%" for lbl in summary)
        print(row, flush=True)

    with open(os.path.join(RESULT_DIR, f"stage1_summary_{args.model}_rehearsal.json"), "w") as f:
        json.dump({"model": args.model, "protocol": "rehearsal", "sweep_points": points, "comparisons": summary}, f, indent=2)

    print(f"\nRehearsal baseline comparison complete for model={args.model}.", flush=True)


if __name__ == "__main__":
    main()
