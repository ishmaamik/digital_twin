"""Synthetic -> real Scenario 1 -> real target rehearsal study.

Models:
  - Random Forest
  - k-NN
  - Fourier-feature k-NN

For each target (scenario32 and scenario32straight), each seed performs:
  1. Fit on the full synthetic Scenario 1 training partition and evaluate on
     the held-out real target partition (synthetic-only zero-shot).
  2. Fit fresh models at each target sample count using 250 synthetic rows,
     250 real Scenario 1 rows, and N real target rows.

Run from digital_twin_extended/:
    python rehearsal_study/run_synth_real_target_nonparametric.py --model rf
    python rehearsal_study/run_synth_real_target_nonparametric.py --model knn
    python rehearsal_study/run_synth_real_target_nonparametric.py --model fourier_knn
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
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")
RESULT_ROOT = "result"
RESULT_PREFIX = "synth_real_target_nonparametric_500cap"
DEFAULT_TARGETS = ("scenario32", "scenario32straight")
MODELS = ("rf", "knn", "fourier_knn")
SWEEP_POINTS = [50, 100, 150, 200]
N_SEEDS = 3
SOURCE_REPLAY_CAP = 500
SYNTH_REPLAY = 250
REAL_REPLAY = 250
NUM_CLASSES = 16
FOURIER_NUM_FREQUENCIES = 6


def paths_for_synthetic():
    return (
        os.path.join(ARCHIVE_DIR, "synth_UE_loc.mat"),
        os.path.join(ARCHIVE_DIR, "synth_beam_power_measured.mat"),
    )


def paths_for_real_scenario1():
    return (
        os.path.join(ARCHIVE_DIR, "ue_relative_pos.mat"),
        os.path.join(ARCHIVE_DIR, "real_beam_pwr.mat"),
    )


def paths_for_target(target):
    return (
        os.path.join(DATA_DIR, f"{target}_ue_relative_pos.mat"),
        os.path.join(DATA_DIR, f"{target}_real_beam_pwr.mat"),
    )


def target_normalization(target, rand_state):
    pos_path, pwr_path = paths_for_target(target)
    pos, _, _ = create_samples(
        pos_path,
        pwr_path,
        rand_state,
        mode="train",
        train_split=1.0,
        num_data_point=None,
        portion=1.0,
    )
    # create_samples applies the default normalization, so read the raw
    # positions directly through its source files is not appropriate here.
    # The existing MLP experiment uses the seven-scenario constants for these
    # comparisons; keep that convention for direct result comparability.
    del pos, pwr_path
    with open(os.path.join(DATA_DIR, "global_normalization_all7.json"), encoding="utf-8") as file:
        norm = json.load(file)
    return norm["max_xy"], norm["max_dist"]


def fourier_encode(features):
    xy = features[:, :2]
    terms = [features]
    for frequency in 2.0 ** np.arange(FOURIER_NUM_FREQUENCIES):
        terms.extend((
            np.sin(xy * frequency * np.pi),
            np.cos(xy * frequency * np.pi),
        ))
    return np.concatenate(terms, axis=-1)


def load_split(paths, rand_state, mode, max_xy, max_dist, num_data_point=None):
    features, labels, power = create_samples(
        paths[0],
        paths[1],
        rand_state,
        mode=mode,
        train_split=0.8,
        num_data_point=num_data_point,
        portion=1.0,
        max_xy=max_xy,
        max_dist=max_dist,
    )
    return features, labels.astype(int), power


def transformed(features, model):
    return fourier_encode(features) if model == "fourier_knn" else features


def classifier_for(model, n_train):
    if model == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    return KNeighborsClassifier(n_neighbors=min(5, max(1, n_train)))


def fit_rehearsal(model, source_features, source_labels, target_features,
                  target_labels, rng):
    source_count = len(source_labels)
    target_count = len(target_labels)
    features = np.concatenate((source_features, target_features))
    labels = np.concatenate((source_labels, target_labels))
    if model == "rf":
        weights = np.concatenate((
            np.full(source_count, 1.0 / source_count),
            np.full(target_count, 1.0 / target_count),
        ))
        return features, labels, weights
    target_indices = rng.integers(0, target_count, size=source_count)
    return (
        np.concatenate((source_features, target_features[target_indices])),
        np.concatenate((source_labels, target_labels[target_indices])),
        None,
    )


def topk_metrics(classifier, features, labels, power):
    probabilities = classifier.predict_proba(features)
    full_probabilities = np.zeros((len(labels), NUM_CLASSES))
    full_probabilities[:, classifier.classes_] = probabilities
    ranking = np.argsort(-full_probabilities, axis=1)
    accuracy = []
    relative_power = []
    true_power = power[np.arange(len(labels)), labels]
    for k in (1, 2, 3, 5):
        topk = ranking[:, :k]
        accuracy.append(np.any(topk == labels[:, None], axis=1).mean())
        predicted_power = np.take_along_axis(power, topk, axis=1).max(axis=1)
        relative_power.append(np.nanmean(predicted_power / true_power))
    return np.asarray(accuracy), np.asarray(relative_power)


def run_target(model, target):
    print(f"\n=== {model}: synthetic -> real1 -> {target} ===", flush=True)
    synthetic_paths = paths_for_synthetic()
    real1_paths = paths_for_real_scenario1()
    target_paths = paths_for_target(target)
    all_accuracy = []
    all_power = []

    for seed_idx in range(N_SEEDS):
        rand_state = 1000 + seed_idx
        rng = np.random.default_rng(rand_state)
        max_xy, max_dist = target_normalization(target, rand_state)

        target_test, target_test_labels, target_test_power = load_split(
            target_paths, rand_state, "test", max_xy, max_dist,
        )
        synthetic_train, synthetic_labels, _ = load_split(
            synthetic_paths, rand_state, "train", max_xy, max_dist,
        )
        real1_train, real1_labels, _ = load_split(
            real1_paths, rand_state, "train", max_xy, max_dist,
        )
        synthetic_train = transformed(synthetic_train, model)
        real1_train = transformed(real1_train, model)
        target_test = transformed(target_test, model)

        classifier = classifier_for(model, len(synthetic_labels))
        classifier.fit(synthetic_train, synthetic_labels)
        seed_accuracy = []
        seed_power = []
        accuracy, power = topk_metrics(
            classifier, target_test, target_test_labels, target_test_power,
        )
        seed_accuracy.append(accuracy)
        seed_power.append(power)

        synthetic_indices = rng.choice(len(synthetic_labels), SYNTH_REPLAY, replace=False)
        real_indices = rng.choice(len(real1_labels), REAL_REPLAY, replace=False)
        replay_features = np.concatenate((
            synthetic_train[synthetic_indices],
            real1_train[real_indices],
        ))
        replay_labels = np.concatenate((
            synthetic_labels[synthetic_indices],
            real1_labels[real_indices],
        ))
        assert len(replay_labels) == SOURCE_REPLAY_CAP

        for target_count in SWEEP_POINTS:
            target_train, target_labels, _ = load_split(
                target_paths,
                rand_state,
                "train",
                max_xy,
                max_dist,
                num_data_point=target_count,
            )
            target_train = transformed(target_train, model)
            features, labels, weights = fit_rehearsal(
                model,
                replay_features,
                replay_labels,
                target_train,
                target_labels,
                rng,
            )
            classifier = classifier_for(model, len(labels))
            if weights is None:
                classifier.fit(features, labels)
            else:
                classifier.fit(features, labels, sample_weight=weights)
            accuracy, power = topk_metrics(
                classifier, target_test, target_test_labels, target_test_power,
            )
            seed_accuracy.append(accuracy)
            seed_power.append(power)

        all_accuracy.append(np.stack(seed_accuracy, axis=-1))
        all_power.append(np.stack(seed_power, axis=-1))
        print(f"  seed {seed_idx} complete", flush=True)

    all_accuracy = np.stack(all_accuracy, axis=-1)
    all_power = np.stack(all_power, axis=-1)
    output_dir = os.path.join(RESULT_ROOT, f"{RESULT_PREFIX}_{model}")
    os.makedirs(output_dir, exist_ok=True)
    base_name = f"stage2_{model}_synth_real1_{target}_rehearsal_500cap"
    savemat(os.path.join(output_dir, f"{base_name}_acc.mat"), {
        "acc": all_accuracy,
        "sweep_points": [0] + SWEEP_POINTS,
    })
    savemat(os.path.join(output_dir, f"{base_name}_pwr.mat"), {
        "pwr": all_power,
        "sweep_points": [0] + SWEEP_POINTS,
    })
    summary = {
        "model": model,
        "target": target,
        "protocol": "synthetic_scenario1_pretrain_then_250_synthetic_plus_250_real_scenario1_plus_N_real_target",
        "source_replay_cap": SOURCE_REPLAY_CAP,
        "sweep_points": [0] + SWEEP_POINTS,
        "n_seeds": N_SEEDS,
        "top2_accuracy": all_accuracy[1].mean(axis=-1).tolist(),
        "top2_relative_power": np.nanmean(all_power[1], axis=-1).tolist(),
    }
    with open(os.path.join(output_dir, f"{base_name}_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=True)
    print(f"Saved {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(DEFAULT_TARGETS),
        choices=["scenario2", "scenario3", "scenario7", "scenario32", "scenario32straight", "scenario33straight", "scenario33"],
    )
    args = parser.parse_args()
    for target in args.targets:
        run_target(args.model, target)
    print(f"Completed synthetic-real-target study for {args.model}.", flush=True)


if __name__ == "__main__":
    main()