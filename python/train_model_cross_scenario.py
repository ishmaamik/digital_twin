"""
Phase 2 / Stage 1: real-to-real environment-shift study.

For each of the four single-factor comparisons defined in the methodology,
train a beam-prediction model from scratch on one DeepSense6G scenario
(the "source"), evaluate it zero-shot on a different scenario (the "target"),
then fine-tune it on an increasing number of real target-scenario samples,
re-evaluating after each step. This reuses the exact same model architecture,
train_model() training loop, and 16-beam labeling already validated in
Stage 0 (see train_model_transfer_learning_measured.py) -- the only new
piece is looping this over scenario PAIRS instead of synthetic-twin -> real,
and using the globally pooled position normalization (data/global_normalization.json)
instead of the Scenario-1-only constants Stage 0 uses.

This script now also runs a second batch, NEW_COMPARISONS: 8 reverse-
direction and new-site comparisons (Scenarios 1/2/4/7/32/33) requested to
extend the original four. These use a separate, all-7-scenario pooled
normalization file (data/global_normalization_all7.json) since they mix
Scenarios 1-4 with the newer 7/32/33 -- the original
data/global_normalization.json (Scenarios 1-4 only) is untouched, and
ORIGINAL_COMPARISONS below is not re-run (its results already exist in
result/ from an earlier run on this branch).

Must be run from the repository root (digital_twin_extended/), e.g.:
    python python/train_model_cross_scenario.py
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

# The original four single-factor comparisons (source, target, label). Kept
# here for reference only -- NOT run by this script anymore, since their
# results already exist in result/ (stage1_time_of_day_at_mcallister_*,
# stage1_time_of_day_at_ruralroad_*, stage1_site_during_day_*,
# stage1_site_during_night_*). See NEW_COMPARISONS below for what this
# script actually runs.
ORIGINAL_COMPARISONS = [
    ("scenario1", "scenario2", "time_of_day_at_mcallister"),  # McAllister: day -> night
    ("scenario3", "scenario4", "time_of_day_at_ruralroad"),   # Rural Road: day -> night
    ("scenario1", "scenario3", "site_during_day"),            # Day: McAllister -> Rural Road
    ("scenario2", "scenario4", "site_during_night"),          # Night: McAllister -> Rural Road
]

# The 8 new comparisons actually run by this script: reverse directions of
# two of the original four, plus McAllister <-> Scenario 7 (wide, day-only),
# McAllister <-> Scenario 32 (narrow, day), and McAllister <-> Scenario 33
# (narrow, night). Labeled s{A}_to_s{B} for consistency with the other
# branches' naming.
NEW_COMPARISONS = [
    ("scenario2", "scenario1", "s2_to_s1"),    # McAllister night -> day (reverse of time_of_day_at_mcallister)
    ("scenario4", "scenario2", "s4_to_s2"),    # Rural Road night -> McAllister night (reverse of site_during_night)
    ("scenario1", "scenario7", "s1_to_s7"),    # McAllister day -> Scenario 7 (wide, day-only)
    ("scenario1", "scenario32", "s1_to_s32"),  # McAllister day -> Scenario 32 (narrow, day)
    ("scenario7", "scenario1", "s7_to_s1"),    # Scenario 7 -> McAllister day (reverse)
    ("scenario32", "scenario1", "s32_to_s1"),  # Scenario 32 -> McAllister day (reverse)
    ("scenario1", "scenario33", "s1_to_s33"),  # McAllister day -> Scenario 33 (narrow, night)
    ("scenario33", "scenario1", "s33_to_s1"),  # Scenario 33 -> McAllister day (reverse)
]

# Diagnostic follow-up: s1_to_s32/s1_to_s33 (and their reverses) showed
# highly unstable, non-monotonic per-seed accuracy in NEW_COMPARISONS above.
# Root-cause investigation (see python/preprocess_college_ave_straight.py)
# found Scenario32/33's road physically bends partway through the captured
# segment (visible in data/scenario32_ue_scatter.png /
# scenario33_ue_scatter.png), unlike every other scenario in this study,
# which are all straight. STRAIGHT_COMPARISONS re-runs the same four
# McAllister<->College Ave pairs using the bend-excluded
# scenario32straight/scenario33straight data (y<40m only, ~65-73% of the
# original samples kept) to test whether removing the bend restores stable,
# monotonic recovery. The original NEW_COMPARISONS results for these four
# pairs (bend included) are NOT overwritten or re-run -- both the "not
# cleaned" (bend-included) and "cleaned" (straight-only) results are kept
# side by side deliberately, since the contrast between them is itself a
# thesis result (showing the bend, not just site/width, drives the
# instability).
STRAIGHT_COMPARISONS = [
    ("scenario1", "scenario32straight", "s1_to_s32straight"),
    ("scenario32straight", "scenario1", "s32straight_to_s1"),
    ("scenario1", "scenario33straight", "s1_to_s33straight"),
    ("scenario33straight", "scenario1", "s33straight_to_s1"),
]

COMPARISONS = STRAIGHT_COMPARISONS

SWEEP_POINTS = list(range(5, 101, 5)) + [150, 200]  # real target samples used for fine-tuning
N_SEEDS = 10  # doubled from the initial 5 after checking per-seed variance on site_during_day

PRETRAIN_EPOCHS = 80
FINETUNE_EPOCHS = 40
PRETRAIN_LR = 1e-2
FINETUNE_LR = 1e-4
BATCH_SIZE = 32
FINETUNE_BATCH_SIZE = 8
VAL_BATCH_SIZE = 128
NUM_CLASSES = 16


def load_global_normalization():
    # Pooled across all 7 scenarios (1-4, 7, 32, 33) -- required since
    # NEW_COMPARISONS mixes Scenarios 1-4 with 7/32/33. Deliberately a
    # separate file from data/global_normalization.json (Scenarios 1-4
    # only), which is left untouched for ORIGINAL_COMPARISONS' reference.
    with open(os.path.join(DATA_DIR, "global_normalization_all7.json")) as f:
        norm = json.load(f)
    return norm["max_xy"], norm["max_dist"]


def paths_for(scenario):
    pos_path = os.path.join(DATA_DIR, f"{scenario}_ue_relative_pos.mat")
    pwr_path = os.path.join(DATA_DIR, f"{scenario}_real_beam_pwr.mat")
    return pos_path, pwr_path


def run_comparison(source, target, label, max_xy, max_dist):
    print(f"\n{'=' * 70}\nComparison: {source} -> {target}  ({label})\n{'=' * 70}", flush=True)

    source_pos, source_pwr = paths_for(source)
    target_pos, target_pwr = paths_for(target)

    all_acc = []  # per seed: (4 metrics, 1 + len(SWEEP_POINTS))
    all_pwr = []

    for seed_idx in range(N_SEEDS):
        rand_state = int(torch.randint(low=1, high=2000, size=(1,)))
        now = datetime.datetime.now().strftime("%H_%M_%S_%f")
        comment = f"crossdomain_{label}_seed{seed_idx}_{now}"

        acc_this_seed = []
        pwr_this_seed = []

        # Held-out target test partition, fixed for this seed across every step below.
        dataset_target_test = DataFeed(pos_path=target_pos, beam_pwr_path=target_pwr,
                                        rand_state=rand_state, mode='test',
                                        max_xy=max_xy, max_dist=max_dist)
        target_test_loader = DataLoader(dataset_target_test, VAL_BATCH_SIZE, shuffle=False)

        # Step A + B: train from scratch on the FULL source training pool,
        # evaluate zero-shot on the target test partition (this IS the
        # "0 real target samples" point in the sweep).
        dataset_source_train = DataFeed(pos_path=source_pos, beam_pwr_path=source_pwr,
                                         rand_state=rand_state, mode='train',
                                         max_xy=max_xy, max_dist=max_dist)
        source_train_loader = DataLoader(dataset_source_train, BATCH_SIZE, shuffle=True)

        print(f"  seed {seed_idx}: training on {source} "
              f"({len(dataset_source_train)} samples), evaluating zero-shot on {target} "
              f"({len(dataset_target_test)} held-out test samples)", flush=True)

        test_loss, test_acc, test_pwr, predictions, raw_predictions, true_label, model_path = train_model(
            train_loader=source_train_loader,
            val_loader=target_test_loader,
            test_loader=target_test_loader,
            comment=comment,
            num_classes=NUM_CLASSES,
            num_epoch=PRETRAIN_EPOCHS,
            if_writer=True,  # must be True: this is what makes train_model() save the checkpoint
            lr=PRETRAIN_LR,
        )
        acc_this_seed.append(test_acc)
        pwr_this_seed.append(test_pwr)

        # Step C: fine-tune on an increasing number of real target-scenario samples,
        # each run starting fresh from the same source-trained checkpoint (not
        # accumulating across sweep points).
        for num_data_point in SWEEP_POINTS:
            dataset_target_train = DataFeed(pos_path=target_pos, beam_pwr_path=target_pwr,
                                             rand_state=rand_state, mode='train',
                                             num_data_point=num_data_point,
                                             max_xy=max_xy, max_dist=max_dist)
            target_train_loader = DataLoader(dataset_target_train, FINETUNE_BATCH_SIZE, shuffle=True)

            test_loss, test_acc, test_pwr, predictions, raw_predictions, true_label, _ = train_model(
                train_loader=target_train_loader,
                val_loader=target_test_loader,
                test_loader=target_test_loader,
                comment=comment,
                num_classes=NUM_CLASSES,
                num_epoch=FINETUNE_EPOCHS,
                if_writer=False,
                model_path=model_path,
                lr=FINETUNE_LR,
            )
            acc_this_seed.append(test_acc)
            pwr_this_seed.append(test_pwr)

        # The checkpoint is only needed transiently to initialize each fine-tuning
        # run above; remove it once this seed's sweep is done.
        if model_path and os.path.exists(model_path):
            os.remove(model_path)

        all_acc.append(np.stack(acc_this_seed, -1))  # (4 metrics, 1+len(SWEEP_POINTS))
        all_pwr.append(np.stack(pwr_this_seed, -1))

    all_acc = np.stack(all_acc, -1)  # (4 metrics, points, seeds)
    all_pwr = np.stack(all_pwr, -1)

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_acc_path = os.path.join(RESULT_DIR, f"stage1_{label}_acc.mat")
    out_pwr_path = os.path.join(RESULT_DIR, f"stage1_{label}_pwr.mat")
    savemat(out_acc_path, {"acc": all_acc, "sweep_points": [0] + SWEEP_POINTS})
    savemat(out_pwr_path, {"pwr": all_pwr, "sweep_points": [0] + SWEEP_POINTS})
    print(f"Saved {out_acc_path} and {out_pwr_path}", flush=True)

    return all_acc, all_pwr


if __name__ == "__main__":
    torch.manual_seed(2022)
    os.makedirs(RESULT_DIR, exist_ok=True)
    max_xy, max_dist = load_global_normalization()
    print(f"Using global normalization: max_xy={max_xy:.4f}, max_dist={max_dist:.4f}", flush=True)

    summary = {}
    for source, target, label in COMPARISONS:
        acc, pwr = run_comparison(source, target, label, max_xy, max_dist)
        top2_acc = acc[1].mean(axis=-1)  # mean over seeds, per sweep point
        top2_pwr = pwr[1].mean(axis=-1)
        summary[label] = {
            "top2_accuracy": top2_acc.tolist(),
            "top2_relative_power": top2_pwr.tolist(),
        }

    points = [0] + SWEEP_POINTS
    print("\n=== Stage 1 summary: mean top-2 accuracy per comparison ===", flush=True)
    header = "real_samples".rjust(12) + "".join(f"{lbl:>28}" for lbl in summary)
    print(header, flush=True)
    for i, p in enumerate(points):
        row = str(p).rjust(12) + "".join(f"{summary[lbl]['top2_accuracy'][i] * 100:>27.2f}%" for lbl in summary)
        print(row, flush=True)

    # Deliberately NOT "stage1_summary.json" or "stage1_summary_new_comparisons.json"
    # -- those already hold the original four and the 8-comparison batch's
    # summaries respectively, and must not be overwritten.
    with open(os.path.join(RESULT_DIR, "stage1_summary_straight_comparisons.json"), "w") as f:
        json.dump({"sweep_points": points, "comparisons": summary}, f, indent=2)

    print("\nStage 1 (straight-segment comparisons) complete.", flush=True)
