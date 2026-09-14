"""
Diagnostic follow-up to Phase 2 / Stage 1: disentangle "more real target
samples" from "more gradient update steps."

train_model() fine-tunes for a FIXED number of epochs with a FIXED batch
size, so increasing num_data_point from 100 to 200 also roughly doubles the
number of gradient updates performed (since there are twice as many batches
per epoch). This script isolates that confound: it fixes num_data_point=100
(the real-data budget) and instead doubles the epoch count (40 -> 80), which
gives approximately the same total number of gradient updates as the
100->200 sample jump (100 samples / 8 batch_size * 80 epochs ~= 200 samples
/ 8 batch_size * 40 epochs ~= 1000 updates), WITHOUT giving the model any
additional real data.

If accuracy at (100 samples, 80 epochs) recovers close to what Phase 2 found
at (200 samples, 40 epochs), that points to optimization budget (gradient
steps) as the main driver. If it stays close to the original (100 samples,
40 epochs) result instead, that points to real data quantity/coverage as the
main driver.

Must be run from the repository root (digital_twin_extended/), e.g.:
    python python/train_model_epoch_ablation.py
"""
import datetime
import json
import os

import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader

from data_feed import DataFeed
from train_model import train_model

DATA_DIR = "data"
RESULT_DIR = "result"

COMPARISONS = [
    ("scenario1", "scenario2", "time_of_day_at_mcallister"),
    ("scenario3", "scenario4", "time_of_day_at_ruralroad"),
    ("scenario1", "scenario3", "site_during_day"),
    ("scenario2", "scenario4", "site_during_night"),
]

FIXED_NUM_DATA_POINT = 100  # real target samples: held fixed for this ablation
FINETUNE_EPOCH_SETTINGS = [40, 80]  # baseline vs. doubled epoch budget (~doubled gradient steps)

PRETRAIN_EPOCHS = 80
PRETRAIN_LR = 1e-2
FINETUNE_LR = 1e-4
BATCH_SIZE = 32
FINETUNE_BATCH_SIZE = 8
VAL_BATCH_SIZE = 128
NUM_CLASSES = 16
N_SEEDS = 5


def load_global_normalization():
    with open(os.path.join(DATA_DIR, "global_normalization.json")) as f:
        norm = json.load(f)
    return norm["max_xy"], norm["max_dist"]


def paths_for(scenario):
    pos_path = os.path.join(DATA_DIR, f"{scenario}_ue_relative_pos.mat")
    pwr_path = os.path.join(DATA_DIR, f"{scenario}_real_beam_pwr.mat")
    return pos_path, pwr_path


def run_comparison(source, target, label, max_xy, max_dist):
    print(f"\n{'=' * 70}\nEpoch ablation: {source} -> {target}  ({label})\n{'=' * 70}", flush=True)

    source_pos, source_pwr = paths_for(source)
    target_pos, target_pwr = paths_for(target)

    # results[epoch_setting] = list of (4,) accuracy arrays, one per seed
    results_acc = {e: [] for e in FINETUNE_EPOCH_SETTINGS}
    results_pwr = {e: [] for e in FINETUNE_EPOCH_SETTINGS}

    for seed_idx in range(N_SEEDS):
        rand_state = int(torch.randint(low=1, high=2000, size=(1,)))
        now = datetime.datetime.now().strftime("%H_%M_%S_%f")
        comment = f"epochablation_{label}_seed{seed_idx}_{now}"

        dataset_target_test = DataFeed(pos_path=target_pos, beam_pwr_path=target_pwr,
                                        rand_state=rand_state, mode='test',
                                        max_xy=max_xy, max_dist=max_dist)
        target_test_loader = DataLoader(dataset_target_test, VAL_BATCH_SIZE, shuffle=False)

        dataset_source_train = DataFeed(pos_path=source_pos, beam_pwr_path=source_pwr,
                                         rand_state=rand_state, mode='train',
                                         max_xy=max_xy, max_dist=max_dist)
        source_train_loader = DataLoader(dataset_source_train, BATCH_SIZE, shuffle=True)

        print(f"  seed {seed_idx}: pretraining on {source}, then fine-tuning on a FIXED "
              f"{FIXED_NUM_DATA_POINT} real {target} samples at two different epoch budgets", flush=True)

        _, _, _, _, _, _, model_path = train_model(
            train_loader=source_train_loader,
            val_loader=target_test_loader,
            test_loader=target_test_loader,
            comment=comment,
            num_classes=NUM_CLASSES,
            num_epoch=PRETRAIN_EPOCHS,
            if_writer=True,
            lr=PRETRAIN_LR,
        )

        # Same fixed real-data sample, same DataLoader, reused for both epoch settings
        # (freshly re-drawn per seed but identical across the two epoch conditions
        # within a seed, since rand_state and num_data_point are unchanged).
        dataset_target_train = DataFeed(pos_path=target_pos, beam_pwr_path=target_pwr,
                                         rand_state=rand_state, mode='train',
                                         num_data_point=FIXED_NUM_DATA_POINT,
                                         max_xy=max_xy, max_dist=max_dist)

        for finetune_epochs in FINETUNE_EPOCH_SETTINGS:
            target_train_loader = DataLoader(dataset_target_train, FINETUNE_BATCH_SIZE, shuffle=True)
            _, test_acc, test_pwr, _, _, _, _ = train_model(
                train_loader=target_train_loader,
                val_loader=target_test_loader,
                test_loader=target_test_loader,
                comment=comment,
                num_classes=NUM_CLASSES,
                num_epoch=finetune_epochs,
                if_writer=False,
                model_path=model_path,  # always fine-tune from the SAME pretrained checkpoint
                lr=FINETUNE_LR,
            )
            results_acc[finetune_epochs].append(test_acc)
            results_pwr[finetune_epochs].append(test_pwr)

        if model_path and os.path.exists(model_path):
            os.remove(model_path)

    summary = {}
    for e in FINETUNE_EPOCH_SETTINGS:
        acc_arr = np.stack(results_acc[e], -1)  # (4 metrics, seeds)
        pwr_arr = np.stack(results_pwr[e], -1)
        summary[e] = {
            "top2_accuracy_mean": float(acc_arr[1].mean()),
            "top2_accuracy_per_seed": acc_arr[1].tolist(),
            "top2_relative_power_mean": float(pwr_arr[1].mean()),
        }

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, f"epoch_ablation_{label}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_path}", flush=True)

    return summary


if __name__ == "__main__":
    torch.manual_seed(2022)
    os.makedirs(RESULT_DIR, exist_ok=True)
    max_xy, max_dist = load_global_normalization()
    print(f"Using global normalization: max_xy={max_xy:.4f}, max_dist={max_dist:.4f}", flush=True)
    print(f"Fixed real target samples: {FIXED_NUM_DATA_POINT}; comparing fine-tune epoch budgets: {FINETUNE_EPOCH_SETTINGS}", flush=True)

    all_summaries = {}
    for source, target, label in COMPARISONS:
        all_summaries[label] = run_comparison(source, target, label, max_xy, max_dist)

    print("\n=== Epoch ablation summary (fixed at 100 real target samples) ===", flush=True)
    print(f"{'comparison':>32}{'40 epochs':>14}{'80 epochs':>14}", flush=True)
    for label, s in all_summaries.items():
        a40 = s[40]["top2_accuracy_mean"] * 100
        a80 = s[80]["top2_accuracy_mean"] * 100
        print(f"{label:>32}{a40:>13.2f}%{a80:>13.2f}%", flush=True)

    with open(os.path.join(RESULT_DIR, "epoch_ablation_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\nEpoch ablation complete.", flush=True)
