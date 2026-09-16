"""500-source-sample-cap rehearsal study for straight-site comparisons.

Runs only KNN, Random Forest, and Fourier-KNN for:
    s1_to_s32straight, s2_to_s33straight,
    s3_to_s32straight, s4_to_s33straight.

The existing full-source straight results are preserved. These outputs use a
cap500 suffix so the two protocols can be compared directly. For each seed,
500 source samples are selected once and reused at zero-shot and every sweep
point. RF gives source and target pools equal aggregate weight; KNN and
Fourier-KNN oversample target rows with replacement to 500 rows.
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
SOURCE_CAP = 500
SWEEP_POINTS = list(range(5, 101, 5)) + [150, 200]
N_SEEDS = 10
NUM_CLASSES = 16
FOURIER_NUM_FREQUENCIES = 6
COMPARISONS = [
    ("scenario1", "scenario32straight", "s1_to_s32straight"),
    ("scenario2", "scenario33straight", "s2_to_s33straight"),
    ("scenario3", "scenario32straight", "s3_to_s32straight"),
    ("scenario4", "scenario33straight", "s4_to_s33straight"),
]


def load_global_normalization():
    with open(os.path.join(DATA_DIR, "global_normalization_all7.json")) as file:
        norm = json.load(file)
    return norm["max_xy"], norm["max_dist"]


def paths_for(scenario):
    return (os.path.join(DATA_DIR, f"{scenario}_ue_relative_pos.mat"),
            os.path.join(DATA_DIR, f"{scenario}_real_beam_pwr.mat"))


def fourier_encode(pos):
    xy = pos[:, :2]
    terms = [pos]
    for frequency in 2.0 ** np.arange(FOURIER_NUM_FREQUENCIES):
        terms.extend((np.sin(xy * frequency * np.pi),
                      np.cos(xy * frequency * np.pi)))
    return np.concatenate(terms, axis=-1)


def load_split(pos_path, pwr_path, rand_state, mode, max_xy, max_dist,
               num_data_point=None, train_split=0.8, fourier=False):
    pos, labels, pwr = create_samples(
        pos_path, pwr_path, rand_state, mode, train_split=train_split,
        num_data_point=num_data_point, portion=1., max_xy=max_xy,
        max_dist=max_dist,
    )
    return (fourier_encode(pos) if fourier else pos), labels, pwr


def classifier_for(model, n_train):
    if model == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=0)
    return KNeighborsClassifier(n_neighbors=min(5, max(1, n_train)))


def metrics(classifier, features, labels, pwr):
    probabilities = classifier.predict_proba(features)
    full_probabilities = np.zeros((features.shape[0], NUM_CLASSES))
    full_probabilities[:, classifier.classes_] = probabilities
    order = np.argsort(-full_probabilities, axis=1)
    max_pwr = pwr[np.arange(len(labels)), labels]
    accuracy = []
    power = []
    for k in (1, 2, 3, 5):
        topk = order[:, :k]
        accuracy.append(np.any(topk == labels[:, None], axis=1).mean())
        topk_pwr = np.take_along_axis(pwr, topk, axis=1).max(axis=1)
        power.append((topk_pwr / max_pwr).mean())
    return np.array(accuracy), np.array(power)


def fit_rehearsal(model, source_features, source_labels, target_features,
                   target_labels, rng):
    source_count = source_features.shape[0]
    target_count = target_features.shape[0]
    if model == "rf":
        features = np.concatenate((source_features, target_features))
        labels = np.concatenate((source_labels, target_labels))
        weights = np.concatenate((
            np.full(source_count, 1.0 / source_count),
            np.full(target_count, 1.0 / target_count),
        ))
        return features, labels, weights
    target_indices = rng.integers(0, target_count, size=source_count)
    features = np.concatenate((source_features, target_features[target_indices]))
    labels = np.concatenate((source_labels, target_labels[target_indices]))
    return features, labels, None


def run_comparison(model, source, target, label, max_xy, max_dist):
    print(f"\n{'=' * 70}\nCap-500 rehearsal [{model}]: {source} -> {target} ({label})\n{'=' * 70}", flush=True)
    source_pos, source_pwr = paths_for(source)
    target_pos, target_pwr = paths_for(target)
    fourier = model == "fourier_knn"
    all_acc = []
    all_pwr = []

    for seed_idx in range(N_SEEDS):
        rand_state = 1000 + seed_idx
        rng = np.random.default_rng(rand_state)
        test_features, test_labels, test_pwr = load_split(
            target_pos, target_pwr, rand_state, "test", max_xy, max_dist,
            fourier=fourier,
        )
        source_features, source_labels, _ = load_split(
            source_pos, source_pwr, rand_state, "train", max_xy, max_dist,
            train_split=1.0, fourier=fourier,
        )
        source_indices = rng.choice(source_features.shape[0], SOURCE_CAP, replace=False)
        source_features = source_features[source_indices]
        source_labels = source_labels[source_indices]

        acc_seed = []
        pwr_seed = []
        classifier = classifier_for(model, source_labels.size)
        classifier.fit(source_features, source_labels)
        acc, pwr = metrics(classifier, test_features, test_labels, test_pwr)
        acc_seed.append(acc)
        pwr_seed.append(pwr)

        for num_data_point in SWEEP_POINTS:
            target_features, target_labels, _ = load_split(
                target_pos, target_pwr, rand_state, "train", max_xy, max_dist,
                num_data_point=num_data_point, fourier=fourier,
            )
            features, labels, weights = fit_rehearsal(
                model, source_features, source_labels, target_features,
                target_labels, rng,
            )
            classifier = classifier_for(model, labels.size)
            if weights is None:
                classifier.fit(features, labels)
            else:
                classifier.fit(features, labels, sample_weight=weights)
            acc, pwr = metrics(classifier, test_features, test_labels, test_pwr)
            acc_seed.append(acc)
            pwr_seed.append(pwr)

        all_acc.append(np.stack(acc_seed, axis=-1))
        all_pwr.append(np.stack(pwr_seed, axis=-1))
        print(f"  seed {seed_idx} complete", flush=True)

    all_acc = np.stack(all_acc, axis=-1)
    all_pwr = np.stack(all_pwr, axis=-1)
    acc_path = os.path.join(RESULT_DIR, f"stage1_{label}_cap500_rehearsal_{model}_acc.mat")
    pwr_path = os.path.join(RESULT_DIR, f"stage1_{label}_cap500_rehearsal_{model}_pwr.mat")
    savemat(acc_path, {"acc": all_acc, "sweep_points": [0] + SWEEP_POINTS})
    savemat(pwr_path, {"pwr": all_pwr, "sweep_points": [0] + SWEEP_POINTS})
    print(f"Saved {acc_path} and {pwr_path}", flush=True)
    return all_acc, all_pwr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["knn", "rf", "fourier_knn"], required=True)
    args = parser.parse_args()
    max_xy, max_dist = load_global_normalization()
    print(f"Using straight data with source cap={SOURCE_CAP}; model={args.model}", flush=True)
    summary = {}
    for source, target, label in COMPARISONS:
        acc, pwr = run_comparison(args.model, source, target, label, max_xy, max_dist)
        summary[label] = {
            "top2_accuracy": acc[1].mean(axis=-1).tolist(),
            "top2_relative_power": pwr[1].mean(axis=-1).tolist(),
        }
    summary_path = os.path.join(RESULT_DIR, f"stage1_summary_{args.model}_rehearsal_straight_cap500.json")
    with open(summary_path, "w") as file:
        json.dump({"model": args.model, "protocol": "rehearsal_straight_source_cap_500",
                   "sweep_points": [0] + SWEEP_POINTS, "comparisons": summary},
                  file, indent=2)
    print(f"Saved {summary_path}", flush=True)
    print(f"Straight cap-500 rehearsal study complete for model={args.model}.", flush=True)


if __name__ == "__main__":
    main()
