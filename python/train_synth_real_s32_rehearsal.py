"""Separate MLP experiment: synthetic -> real Scenario 1 -> Scenario 32.

Protocol per seed:
  1. Pretrain from scratch on the full Scenario-1 synthetic digital-twin
     dataset in data/archive/synth_*. This creates the common checkpoint.
  2. For each N in (50, 100, 150, 200), start from that checkpoint and
     fine-tune on a rehearsal set containing 250 synthetic Scenario-1 rows,
     250 real Scenario-1 training rows, and N Scenario-32 training rows.
     The 500-sample cap applies to the combined Scenario-1 synthetic+real
     replay pool, and the two source domains are balanced deliberately.
  3. Evaluate on the fixed held-out Scenario-32 test partition.

Scenario-32 training-only normalization is computed once per seed from the
Scenario-32 training pool and applied to synthetic Scenario 1, real Scenario
1, Scenario 32 train, and Scenario 32 test. No Scenario-32 test labels are
used for training, normalization, or model selection.

Each N starts from the same synthetic-pretrained checkpoint; runs do not
chain from N=50 to N=100. Outputs are isolated under result/synth_real_s32_rehearsal_500cap/.
Run from digital_twin_extended/:
    python python/train_synth_real_s32_rehearsal.py
"""
import datetime
import json
import os

import numpy as np
import torch
from scipy.io import loadmat, savemat
from torch.utils.data import ConcatDataset, DataLoader, Subset

from data_feed import DataFeed
from train_model import train_model

DATA_DIR = "data"
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")
TARGET_SCENARIO = os.environ.get("SYNTH_REAL_TARGET_SCENARIO", "scenario32")
TARGET_LABEL = TARGET_SCENARIO.replace("scenario", "s")
RESULT_DIR = os.path.join("result", f"synth_real_{TARGET_LABEL}_rehearsal_500cap_small")
SWEEP_POINTS = [50, 100, 150, 200]
N_SEEDS = 3
SYNTH_REPLAY = 250
REAL_REPLAY = 250
PRETRAIN_EPOCHS = 40
FINETUNE_EPOCHS = 20
PRETRAIN_LR = 1e-2
FINETUNE_LR = 1e-4
PRETRAIN_BATCH_SIZE = 32
FINETUNE_BATCH_SIZE = 8
TEST_BATCH_SIZE = 128
NUM_CLASSES = 16


def target_normalization(rand_state):
    """Compute Scenario-32 scaling from its training partition only."""
    pos_path = os.path.join(DATA_DIR, f"{TARGET_SCENARIO}_ue_relative_pos.mat")
    pwr_path = os.path.join(DATA_DIR, f"{TARGET_SCENARIO}_real_beam_pwr.mat")
    pos = loadmat(pos_path)["ue_relative_pos"]
    pwr = loadmat(pwr_path)["real_beam_pwr"]
    order = np.random.default_rng(rand_state).permutation(len(pos))
    train_count = int(len(pos) * 0.8)
    train_pos = pos[order[:train_count]]
    max_xy = float(np.max(np.abs(train_pos)))
    max_dist = float(np.max(np.sqrt(np.sum(train_pos ** 2, axis=1))))
    return max_xy, max_dist


def dataset(path_pair, rand_state, mode, max_xy, max_dist, num_data_point=None,
            train_split=0.8):
    return DataFeed(
        pos_path=path_pair[0], beam_pwr_path=path_pair[1], rand_state=rand_state,
        mode=mode, num_data_point=num_data_point, train_split=train_split,
        max_xy=max_xy, max_dist=max_dist,
    )


def paths():
    return {
        "synth": (os.path.join(ARCHIVE_DIR, "synth_UE_loc.mat"),
                  os.path.join(ARCHIVE_DIR, "synth_beam_power_measured.mat")),
        "real1": (os.path.join(ARCHIVE_DIR, "ue_relative_pos.mat"),
                  os.path.join(ARCHIVE_DIR, "real_beam_pwr.mat")),
        "target": (os.path.join(DATA_DIR, f"{TARGET_SCENARIO}_ue_relative_pos.mat"),
                   os.path.join(DATA_DIR, f"{TARGET_SCENARIO}_real_beam_pwr.mat")),
        "s32": (os.path.join(DATA_DIR, f"{TARGET_SCENARIO}_ue_relative_pos.mat"),
                os.path.join(DATA_DIR, f"{TARGET_SCENARIO}_real_beam_pwr.mat")),
    }


def run_seed(seed_idx, source_paths):
    rand_state = 1000 + seed_idx
    max_xy, max_dist = target_normalization(rand_state)
    target_test = dataset(source_paths["target"], rand_state, "test", max_xy, max_dist)
    target_test_loader = DataLoader(target_test, TEST_BATCH_SIZE, shuffle=False)

    synth_full = dataset(source_paths["synth"], rand_state, "train", max_xy, max_dist)
    synth_loader = DataLoader(synth_full, PRETRAIN_BATCH_SIZE, shuffle=True)
    comment = f"synth_real_{TARGET_LABEL}_cap500_seed{seed_idx}_{datetime.datetime.now():%H_%M_%S_%f}"
    _, zero_acc, zero_pwr, _, _, _, model_path = train_model(
        train_loader=synth_loader,
        val_loader=target_test_loader,
        test_loader=target_test_loader,
        comment=comment,
        num_classes=NUM_CLASSES,
        num_epoch=PRETRAIN_EPOCHS,
        if_writer=True,
        lr=PRETRAIN_LR,
    )

    # Draw a fixed balanced 500-row replay buffer once per seed.
    synth_replay = Subset(synth_full, torch.randperm(len(synth_full))[:SYNTH_REPLAY].tolist())
    real_full = dataset(source_paths["real1"], rand_state, "train", max_xy, max_dist)
    real_replay = Subset(real_full, torch.randperm(len(real_full))[:REAL_REPLAY].tolist())
    acc_points = [zero_acc]
    pwr_points = [zero_pwr]

    for n_target in SWEEP_POINTS:
        target_train = dataset(source_paths["target"], rand_state, "train", max_xy, max_dist,
                                num_data_point=n_target)
        combined = ConcatDataset([synth_replay, real_replay, target_train])
        combined_loader = DataLoader(combined, FINETUNE_BATCH_SIZE, shuffle=True)
        _, acc, pwr, _, _, _, _ = train_model(
            train_loader=combined_loader,
            val_loader=target_test_loader,
            test_loader=target_test_loader,
            comment=comment,
            num_classes=NUM_CLASSES,
            num_epoch=FINETUNE_EPOCHS,
            if_writer=False,
            model_path=model_path,
            lr=FINETUNE_LR,
        )
        acc_points.append(acc)
        pwr_points.append(pwr)

    if model_path and os.path.exists(model_path):
        os.remove(model_path)
    return np.stack(acc_points, axis=-1), np.stack(pwr_points, axis=-1), (max_xy, max_dist)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    source_paths = paths()
    all_acc, all_pwr, norms = [], [], []
    for seed_idx in range(N_SEEDS):
        print(f"seed {seed_idx}: synthetic pretrain -> Scenario 1 replay -> {TARGET_SCENARIO}", flush=True)
        acc, pwr, norm = run_seed(seed_idx, source_paths)
        all_acc.append(acc)
        all_pwr.append(pwr)
        norms.append(norm)
        print(f"seed {seed_idx} complete", flush=True)

    all_acc = np.stack(all_acc, axis=-1)
    all_pwr = np.stack(all_pwr, axis=-1)
    savemat(os.path.join(RESULT_DIR, f"stage2_synth_real_{TARGET_LABEL}_rehearsal_500cap_acc.mat"),
            {"acc": all_acc, "sweep_points": [0] + SWEEP_POINTS})
    savemat(os.path.join(RESULT_DIR, f"stage2_synth_real_{TARGET_LABEL}_rehearsal_500cap_pwr.mat"),
            {"pwr": all_pwr, "sweep_points": [0] + SWEEP_POINTS})
    summary = {
        "target": TARGET_SCENARIO,
        "protocol": "full_scenario1_synthetic_pretrain_then_250_synthetic_plus_250_real_scenario1_plus_N_real_target",
        "source_replay_cap": 500,
        "sweep_points": [0] + SWEEP_POINTS,
        "normalization": f"{TARGET_SCENARIO} training partition per seed",
        "top2_accuracy": all_acc[1].mean(axis=-1).tolist(),
        "top2_relative_power": all_pwr[1].mean(axis=-1).tolist(),
        "normalization_values": norms,
    }
    with open(os.path.join(RESULT_DIR, f"stage2_synth_real_{TARGET_LABEL}_rehearsal_500cap_summary.json"), "w") as file:
        json.dump(summary, file, indent=2)
    print("synth-real-Scenario32 rehearsal study complete.", flush=True)


if __name__ == "__main__":
    main()
