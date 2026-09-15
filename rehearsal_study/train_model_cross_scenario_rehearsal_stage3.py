"""
Rehearsal-protocol Stage-2 study, batch 3: TinyMLP (neural model side).

Sibling of train_baselines_cross_scenario_rehearsal_stage3.py, which
covers knn/rf/fourier_knn/fourier_rf on the SAME 8 comparisons:

    7 -> 2, 7 -> 3, 7 -> 4, 7 -> 32straight,
    32straight -> 1, 32straight -> 2, 32straight -> 3, 32straight -> 33straight

Both scripts use the SAME REHEARSAL protocol and write results with the
SAME "_rehearsal" naming convention, so results are directly comparable
across all 5 models. Kept fully self-contained (its own
COMPARISONS/SWEEP_POINTS/N_SEEDS) and with its own "_stage3" summary
filename, matching the precedent set by the sibling scripts (see
train_baselines_cross_scenario_rehearsal.py and _stage3.py's docstrings
for why the summary.json filename must not be shared between batches).

REHEARSAL PROTOCOL, per seed (replaces the old target-only-fine-tune
protocol used by python/train_model_cross_scenario.py -- see the sibling
scripts' docstrings for the full history):
  1. Zero-shot (N=0): train TinyMLP from scratch on the FULL source
     scenario dataset (train_split=1.0 -- all of it, no held-out split,
     since the source is training input only and is never evaluated).
     Evaluate zero-shot on the target scenario's held-out test partition
     (80/20 split, mode='test'), fixed once per seed and reused for every
     sweep point below. This step also produces the checkpoint used to
     initialize every fine-tuning run in step 2.
  2. For each N in SWEEP_POINTS: draw N samples from the target scenario's
     80% train pool. Fine-tune FROM the step-1 checkpoint (not from
     scratch, and not accumulating across sweep points -- every sweep
     point starts from the same zero-shot checkpoint) on the REHEARSAL set
     = the full source dataset UNION these N target samples, then evaluate
     on the same fixed target test partition.

Source/target imbalance -- EQUAL AGGREGATE WEIGHT scheme (unchanged from
the sibling TinyMLP script): the combined fine-tuning dataset is wrapped
in a WeightedRandomSampler with per-sample weight 1/n_source for source
rows and 1/N for target rows, and num_samples equal to the combined
dataset size, so each fine-tuning epoch draws from source and target with
equal total probability mass regardless of how tiny N is.

NOTE on Scenario 32straight/33straight data quality: same caveat as the
sibling baseline script -- data/scenario32straight_real_beam_pwr.mat and
scenario33straight_real_beam_pwr.mat each contain a small number of
pre-existing NaN cells (confirmed present in already-committed
non-rehearsal results too, not introduced by any rehearsal script). This
script runs unmodified against that data with no special-casing per
explicit instruction. When 32straight or 33straight is the TARGET
(7->32straight, 32straight->33straight here), a small number of seeds'
"pwr" metric may come out NaN; "acc" is unaffected. Expected, not a bug.

COMPUTE WARNING: every one of the 23 sweep points (22 SWEEP_POINTS + the
zero-shot point) fine-tunes over a resampled set the size of n_source + N
for FINETUNE_EPOCHS=40 epochs. Across N_SEEDS=10 seeds and 8 comparisons
this is the heaviest of the 5 models here -- budget accordingly. Check for
"Stage 2 (TinyMLP rehearsal, batch 3) complete." in the log, not just file
existence, before treating a run as finished.

Must be run from the repository root, e.g.:
    python rehearsal_study/train_model_cross_scenario_rehearsal_stage3.py --model tinymlp
"""
import argparse
import datetime
import json
import os

import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from data_feed import DataFeed
from model import MODEL_REGISTRY
from train_model import train_model

DATA_DIR = "data"
RESULT_DIR = "result"

COMPARISONS = [
    ("scenario7", "scenario2", "s7_to_s2"),
    ("scenario7", "scenario3", "s7_to_s3"),
    ("scenario7", "scenario4", "s7_to_s4"),
    ("scenario7", "scenario32straight", "s7_to_s32straight"),
    ("scenario32straight", "scenario1", "s32straight_to_s1"),
    ("scenario32straight", "scenario2", "s32straight_to_s2"),
    ("scenario32straight", "scenario3", "s32straight_to_s3"),
    ("scenario32straight", "scenario33straight", "s32straight_to_s33straight"),
]

SWEEP_POINTS = list(range(5, 101, 5)) + [150, 200]  # real target samples used for fine-tuning
N_SEEDS = 10

PRETRAIN_EPOCHS = 80
FINETUNE_EPOCHS = 40
PRETRAIN_LR = 1e-2
FINETUNE_LR = 1e-4
BATCH_SIZE = 32
FINETUNE_BATCH_SIZE = 8
VAL_BATCH_SIZE = 128
NUM_CLASSES = 16


def load_global_normalization():
    with open(os.path.join(DATA_DIR, "global_normalization_all7.json")) as f:
        norm = json.load(f)
    return norm["max_xy"], norm["max_dist"]


def paths_for(scenario):
    pos_path = os.path.join(DATA_DIR, f"{scenario}_ue_relative_pos.mat")
    pwr_path = os.path.join(DATA_DIR, f"{scenario}_real_beam_pwr.mat")
    return pos_path, pwr_path


def build_rehearsal_loader(dataset_source_train, dataset_target_train, batch_size):
    """Combine the full source dataset with the N-sample target dataset and
    wrap them in a WeightedRandomSampler giving the target block equal
    aggregate sampling weight to the source pool (see module docstring)."""
    n_source = len(dataset_source_train)
    n_target = len(dataset_target_train)
    combined = ConcatDataset([dataset_source_train, dataset_target_train])
    weights = np.concatenate([
        np.full(n_source, 1.0 / n_source),
        np.full(n_target, 1.0 / n_target),
    ])
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(combined),
        replacement=True,
    )
    return DataLoader(combined, batch_size=batch_size, sampler=sampler)


def run_comparison(source, target, label, max_xy, max_dist, model_name='tinymlp'):
    print(f"\n{'=' * 70}\nRehearsal comparison [{model_name}]: {source} -> {target}  ({label})\n{'=' * 70}", flush=True)

    source_pos, source_pwr = paths_for(source)
    target_pos, target_pwr = paths_for(target)

    all_acc = []  # per seed: (4 metrics, 1 + len(SWEEP_POINTS))
    all_pwr = []

    for seed_idx in range(N_SEEDS):
        rand_state = int(torch.randint(low=1, high=2000, size=(1,)))
        now = datetime.datetime.now().strftime("%H_%M_%S_%f")
        comment = f"rehearsal_stage3_{label}_seed{seed_idx}_{now}"

        acc_this_seed = []
        pwr_this_seed = []

        # Held-out target test partition, fixed for this seed across every step below.
        dataset_target_test = DataFeed(pos_path=target_pos, beam_pwr_path=target_pwr,
                                        rand_state=rand_state, mode='test',
                                        max_xy=max_xy, max_dist=max_dist)
        target_test_loader = DataLoader(dataset_target_test, VAL_BATCH_SIZE, shuffle=False)

        # Step 1: train from scratch on the FULL source dataset (all of it --
        # train_split=1.0, since the source is training input only), evaluate
        # zero-shot on the target test partition. This is the "0 real target
        # samples" point in the sweep and also produces the checkpoint used
        # to initialize every fine-tuning run below.
        dataset_source_train = DataFeed(pos_path=source_pos, beam_pwr_path=source_pwr,
                                         rand_state=rand_state, mode='train', train_split=1.0,
                                         max_xy=max_xy, max_dist=max_dist)
        source_train_loader = DataLoader(dataset_source_train, BATCH_SIZE, shuffle=True)

        print(f"  seed {seed_idx}: pretraining on FULL {source} "
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
            model_name=model_name,
        )
        acc_this_seed.append(test_acc)
        pwr_this_seed.append(test_pwr)

        # Step 2: rehearsal sweep -- fine-tune FROM the step-1 checkpoint on
        # (full source) UNION (N target samples), re-evaluating each time on
        # the same fixed target test partition. Every sweep point restarts
        # from the SAME step-1 checkpoint (train_model does not overwrite it
        # when if_writer=False), so this does not accumulate across points.
        for num_data_point in SWEEP_POINTS:
            dataset_target_train = DataFeed(pos_path=target_pos, beam_pwr_path=target_pwr,
                                             rand_state=rand_state, mode='train',
                                             num_data_point=num_data_point,
                                             max_xy=max_xy, max_dist=max_dist)
            rehearsal_loader = build_rehearsal_loader(
                dataset_source_train, dataset_target_train, FINETUNE_BATCH_SIZE,
            )

            test_loss, test_acc, test_pwr, predictions, raw_predictions, true_label, _ = train_model(
                train_loader=rehearsal_loader,
                val_loader=target_test_loader,
                test_loader=target_test_loader,
                comment=comment,
                num_classes=NUM_CLASSES,
                num_epoch=FINETUNE_EPOCHS,
                if_writer=False,
                model_path=model_path,
                lr=FINETUNE_LR,
                model_name=model_name,
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
    out_acc_path = os.path.join(RESULT_DIR, f"stage1_{label}_rehearsal_{model_name}_acc.mat")
    out_pwr_path = os.path.join(RESULT_DIR, f"stage1_{label}_rehearsal_{model_name}_pwr.mat")
    savemat(out_acc_path, {"acc": all_acc, "sweep_points": [0] + SWEEP_POINTS})
    savemat(out_pwr_path, {"pwr": all_pwr, "sweep_points": [0] + SWEEP_POINTS})
    print(f"Saved {out_acc_path} and {out_pwr_path}", flush=True)

    return all_acc, all_pwr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="tinymlp",
                         help="Which architecture to run the Stage 2 rehearsal protocol with. "
                              "This study only tracks tinymlp here; plain mlp is tracked separately.")
    args = parser.parse_args()

    torch.manual_seed(2022)
    os.makedirs(RESULT_DIR, exist_ok=True)
    max_xy, max_dist = load_global_normalization()
    print(f"Using global normalization: max_xy={max_xy:.4f}, max_dist={max_dist:.4f}", flush=True)
    print(f"Running Stage 2 REHEARSAL (batch 3) with model={args.model}", flush=True)

    summary = {}
    for source, target, label in COMPARISONS:
        acc, pwr = run_comparison(source, target, label, max_xy, max_dist, model_name=args.model)
        top2_acc = acc[1].mean(axis=-1)  # mean over seeds, per sweep point
        top2_pwr = pwr[1].mean(axis=-1)
        summary[label] = {
            "top2_accuracy": top2_acc.tolist(),
            "top2_relative_power": top2_pwr.tolist(),
        }

    points = [0] + SWEEP_POINTS
    print(f"\n=== Stage 2 REHEARSAL batch-3 summary [{args.model}]: mean top-2 accuracy per comparison ===", flush=True)
    header = "real_samples".rjust(12) + "".join(f"{lbl:>24}" for lbl in summary)
    print(header, flush=True)
    for i, p in enumerate(points):
        row = str(p).rjust(12) + "".join(f"{summary[lbl]['top2_accuracy'][i] * 100:>23.2f}%" for lbl in summary)
        print(row, flush=True)

    with open(os.path.join(RESULT_DIR, f"stage1_summary_{args.model}_rehearsal_stage3.json"), "w") as f:
        json.dump({"model": args.model, "protocol": "rehearsal", "sweep_points": points, "comparisons": summary}, f, indent=2)

    print(f"\nStage 2 (TinyMLP rehearsal, batch 3) complete.", flush=True)
