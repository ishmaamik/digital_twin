"""
Rehearsal-based MLP training, batch 3: 10 comparisons anchored on Scenario 7
(wide site) and the College-Ave-straight sites (32straight/33straight) as
both source and target, MLP ONLY (no knn/rf/fourier_knn/fourier_rf --
explicitly out of scope for this batch per instruction).

Run as its own separate process, in parallel with (not instead of) the
already-running sibling batch in train_model_rehearsal_extended.py (which
covers s1_to_s32straight / s1_to_s33straight). To avoid a race condition
where both processes' incremental load-merge-write of a SHARED summary JSON
could clobber each other's newly-written entries, this script writes its
own separate summary file (stage1_summary_rehearsal_extended_batch3.json)
instead of stage1_summary_rehearsal_extended.json. Per-comparison .mat
files still go to the same result/ directory as every other batch, which is
safe since every comparison label in this batch is unique.

Unlike every previous MLP script in this project (train_model_cross_scenario.py),
which continues fine-tuning on N target-only samples at each sweep point, this
script implements true rehearsal (Jiang & Alkhateeb's CSI rehearsal technique,
cited in the thesis as jiang2024digitaltwincsi -- "Paper B"): at every sweep
point, the model is fine-tuned on the FULL source training pool COMBINED WITH
the N real target samples for that point, rather than on the N target samples
alone. This is meant to avoid catastrophic forgetting of the source-domain
rule during fine-tuning.

Protocol per (source, target) comparison, per seed:
  1. Zero-shot: train from scratch on the full source training pool (80
     epochs, PRETRAIN_LR), evaluate on the target's held-out test partition.
     This is the "0 real target samples" point -- identical in spirit to
     every other MLP comparison in this project.
  2. For each N in SWEEP_POINTS: draw N samples from the target's training
     pool, build a combined dataset = full source training pool UNION these
     N target samples (torch ConcatDataset), and continue fine-tuning FROM
     the zero-shot checkpoint on this combined set (40 epochs, FINETUNE_LR).
     Evaluate on the same held-out target test partition.

Because every sweep point now retrains on a dataset close to the full source
size rather than on a handful of target samples, this is far more expensive
per comparison than the earlier cross-scenario scripts -- see the time
estimate given before this script was run.

Uses the 7-scenario pooled normalization (data/global_normalization_all7.json),
matching every other script in this project that mixes Scenarios 1-4 with
7/32/33. Output files are suffixed "_rehearsal" so nothing collides with any
existing (non-rehearsal) result for scenarios that already appear in other
comparisons.

Robustness: each (comparison, seed) unit is wrapped in its own try/except so
a single failure cannot take down the rest of this multi-hour run. Failures
are logged clearly and the affected seed is skipped; partial per-comparison
results are still saved and averaged over however many seeds succeeded.

Must be run from the repository root (digital_twin_extended/):
    python python/train_model_rehearsal_extended.py
"""
import datetime
import json
import os
import traceback

import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import ConcatDataset, DataLoader, Subset

from data_feed import DataFeed
from train_model import train_model
from train_model_cross_scenario import load_global_normalization, paths_for

DATA_DIR = "data"
RESULT_DIR = "result"


# Every pair involving Scenario 32 or 33 uses the straight-segment-only data
# (scenario32straight / scenario33straight, y<40m, bend excluded -- see
# python/preprocess_college_ave_straight.py) rather than the full/bend-
# included scenario32/scenario33 data, per explicit instruction. Pairs not
# involving 32/33 (e.g. scenario2<->scenario7, scenario3<->scenario1) are
# unaffected and use the original full scenario data as before.
COMPARISONS = [
    ("scenario7", "scenario3", "s7_to_s3"),
    ("scenario7", "scenario4", "s7_to_s4"),
    ("scenario7", "scenario32straight", "s7_to_s32straight"),
    ("scenario32straight", "scenario2", "s32straight_to_s2"),
    ("scenario32straight", "scenario3", "s32straight_to_s3"),
    ("scenario32straight", "scenario7", "s32straight_to_s7"),
    ("scenario32straight", "scenario33straight", "s32straight_to_s33straight"),
    ("scenario33straight", "scenario2", "s33straight_to_s2"),
    ("scenario33straight", "scenario4", "s33straight_to_s4"),
    ("scenario33straight", "scenario32straight", "s33straight_to_s32straight"),
]

SWEEP_POINTS = list(range(5, 101, 5)) + [150, 200]
N_SEEDS = 10

PRETRAIN_EPOCHS = 80
FINETUNE_EPOCHS = 40
PRETRAIN_LR = 1e-2
FINETUNE_LR = 1e-4
BATCH_SIZE = 32
FINETUNE_BATCH_SIZE = 8
VAL_BATCH_SIZE = 128
NUM_CLASSES = 16

# Cap on how much of the source pool is replayed during REHEARSAL fine-tuning
# (Step 2 of the protocol below). Replaying the full source pool (up to ~2400
# samples) at every one of the 23 sweep points, for every seed, is what was
# making this batch take ~13 minutes/seed -- most of that cost is retraining
# on source rows that don't change sweep point to sweep point. This caps the
# rehearsal replay to a fixed-size random subset of the source pool (drawn
# once per seed, reused across that seed's whole sweep), while leaving the
# ZERO-SHOT step (the initial "train on scratch on the full source" call)
# completely untouched, since that number needs to stay comparable to every
# other zero-shot result in this project, which all use the full source pool.
REHEARSAL_SOURCE_CAP = 500

N_METRICS = 4  # top-1, top-2, top-3, top-5, matching every other script's acc/pwr shape
N_POINTS = 1 + len(SWEEP_POINTS)


def run_comparison(source, target, label, max_xy, max_dist):
    print(f"\n{'=' * 70}\nRehearsal comparison: {source} -> {target}  ({label})\n{'=' * 70}", flush=True)

    source_pos, source_pwr = paths_for(source)
    target_pos, target_pwr = paths_for(target)

    all_acc = []
    all_pwr = []
    n_ok = 0

    for seed_idx in range(N_SEEDS):
        try:
            rand_state = int(torch.randint(low=1, high=2000, size=(1,)))
            now = datetime.datetime.now().strftime("%H_%M_%S_%f")
            comment = f"rehearsal_{label}_seed{seed_idx}_{now}"

            acc_this_seed = []
            pwr_this_seed = []

            dataset_target_test = DataFeed(pos_path=target_pos, beam_pwr_path=target_pwr,
                                            rand_state=rand_state, mode='test',
                                            max_xy=max_xy, max_dist=max_dist)
            target_test_loader = DataLoader(dataset_target_test, VAL_BATCH_SIZE, shuffle=False)

            dataset_source_train = DataFeed(pos_path=source_pos, beam_pwr_path=source_pwr,
                                             rand_state=rand_state, mode='train',
                                             max_xy=max_xy, max_dist=max_dist)
            source_train_loader = DataLoader(dataset_source_train, BATCH_SIZE, shuffle=True)

            print(f"  seed {seed_idx}: pretrain on {source} ({len(dataset_source_train)} samples), "
                  f"zero-shot eval on {target} ({len(dataset_target_test)} held-out test samples)", flush=True)

            test_loss, test_acc, test_pwr, predictions, raw_predictions, true_label, model_path = train_model(
                train_loader=source_train_loader,
                val_loader=target_test_loader,
                test_loader=target_test_loader,
                comment=comment,
                num_classes=NUM_CLASSES,
                num_epoch=PRETRAIN_EPOCHS,
                if_writer=True,
                lr=PRETRAIN_LR,
            )
            acc_this_seed.append(test_acc)
            pwr_this_seed.append(test_pwr)

            # Fixed-size rehearsal replay buffer for this seed: a random cap-
            # sized subset of the full source pool, drawn once and reused at
            # every sweep point below (not the full source pool used above).
            if len(dataset_source_train) > REHEARSAL_SOURCE_CAP:
                g = torch.Generator().manual_seed(rand_state)
                cap_idx = torch.randperm(len(dataset_source_train), generator=g)[:REHEARSAL_SOURCE_CAP].tolist()
                rehearsal_source = Subset(dataset_source_train, cap_idx)
            else:
                rehearsal_source = dataset_source_train
            print(f"    rehearsal replay buffer: {len(rehearsal_source)} of "
                  f"{len(dataset_source_train)} source samples", flush=True)

            for num_data_point in SWEEP_POINTS:
                dataset_target_train = DataFeed(pos_path=target_pos, beam_pwr_path=target_pwr,
                                                 rand_state=rand_state, mode='train',
                                                 num_data_point=num_data_point,
                                                 max_xy=max_xy, max_dist=max_dist)
                # Rehearsal: fine-tune on the capped source replay buffer +
                # N target samples combined, not on the N target samples alone.
                combined_dataset = ConcatDataset([rehearsal_source, dataset_target_train])
                combined_loader = DataLoader(combined_dataset, FINETUNE_BATCH_SIZE, shuffle=True)

                test_loss, test_acc, test_pwr, predictions, raw_predictions, true_label, _ = train_model(
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
                acc_this_seed.append(test_acc)
                pwr_this_seed.append(test_pwr)

            if model_path and os.path.exists(model_path):
                os.remove(model_path)

            all_acc.append(np.stack(acc_this_seed, -1))
            all_pwr.append(np.stack(pwr_this_seed, -1))
            n_ok += 1

        except Exception:
            print(f"  !! seed {seed_idx} FAILED for {label} -- skipping this seed, continuing.", flush=True)
            traceback.print_exc()
            continue

    if n_ok == 0:
        print(f"  !! ALL seeds failed for {label} -- no result saved for this comparison.", flush=True)
        return None, None

    all_acc = np.stack(all_acc, -1)
    all_pwr = np.stack(all_pwr, -1)

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_acc_path = os.path.join(RESULT_DIR, f"stage1_{label}_rehearsal_acc.mat")
    out_pwr_path = os.path.join(RESULT_DIR, f"stage1_{label}_rehearsal_pwr.mat")
    savemat(out_acc_path, {"acc": all_acc, "sweep_points": [0] + SWEEP_POINTS, "n_seeds_completed": n_ok})
    savemat(out_pwr_path, {"pwr": all_pwr, "sweep_points": [0] + SWEEP_POINTS, "n_seeds_completed": n_ok})
    print(f"Saved {out_acc_path} and {out_pwr_path} ({n_ok}/{N_SEEDS} seeds completed)", flush=True)

    return all_acc, all_pwr


if __name__ == "__main__":
    torch.manual_seed(2022)
    os.makedirs(RESULT_DIR, exist_ok=True)
    max_xy, max_dist = load_global_normalization()
    print(f"Using global (7-scenario) normalization: max_xy={max_xy:.4f}, max_dist={max_dist:.4f}", flush=True)
    print(f"Running {len(COMPARISONS)} rehearsal comparisons, {N_SEEDS} seeds each.", flush=True)

    # Separate summary file from the sibling batch3-vs-batch1 script (see
    # module docstring) -- avoids a read-modify-write race between two
    # concurrently running processes on the same JSON file.
    summary_path = os.path.join(RESULT_DIR, "stage1_summary_rehearsal_extended_batch3.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f).get("comparisons", {})

    for source, target, label in COMPARISONS:
        try:
            acc, pwr = run_comparison(source, target, label, max_xy, max_dist)
        except Exception:
            print(f"!! Comparison {label} FAILED entirely -- skipping, continuing to next comparison.", flush=True)
            traceback.print_exc()
            continue

        if acc is None:
            continue

        top2_acc = acc[1].mean(axis=-1)
        top2_pwr = pwr[1].mean(axis=-1)
        summary[label] = {
            "top2_accuracy": top2_acc.tolist(),
            "top2_relative_power": top2_pwr.tolist(),
        }

        # Write the summary incrementally after every comparison, not only at
        # the very end, so a multi-hour run's progress is never lost if it is
        # interrupted partway through.
        points = [0] + SWEEP_POINTS
        with open(summary_path, "w") as f:
            json.dump({"sweep_points": points, "comparisons": summary}, f, indent=2)

    print("\n=== Rehearsal-extended batch-3 summary: mean top-2 accuracy per comparison ===", flush=True)
    points = [0] + SWEEP_POINTS
    for lbl, s in summary.items():
        print(f"{lbl:16s} 0={s['top2_accuracy'][0]*100:6.2f}%  "
              f"100={s['top2_accuracy'][20]*100:6.2f}%  200={s['top2_accuracy'][-1]*100:6.2f}%", flush=True)

    print(f"\nCompleted {len(summary)}/{len(COMPARISONS)} comparisons successfully.", flush=True)
    print("Rehearsal-extended batch-3 complete.", flush=True)
