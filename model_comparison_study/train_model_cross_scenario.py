"""
Phase 2 / Stage 1: real-to-real environment-shift study.

For each of the single-factor comparisons defined in the methodology,
train a beam-prediction model from scratch on one DeepSense6G scenario
(the "source"), evaluate it zero-shot on a different scenario (the "target"),
then fine-tune it on an increasing number of real target-scenario samples,
re-evaluating after each step. This reuses the exact same model architecture,
train_model() training loop, and 16-beam labeling already validated in
Stage 0 (see train_model_transfer_learning_measured.py) -- the only new
piece is looping this over scenario PAIRS instead of synthetic-twin -> real.

This script now runs a separate, second batch of comparisons -- 6 new
lane-width comparisons, involving 3 additional scenarios: 32/33 (a second
day/night pair at a site with the same 2-lane width as McAllister) and 7 (a
single wide, 4-lane site). These isolate whether site-shift severity tracks
a difference in lane width specifically, using same-width transfer
(1->32, 2->33), two independent narrow->wide replications (1->7, 32->7),
and a wide->wide control (3->7).

The original 4 single-factor comparisons among Scenarios 1-4 (McAllister
Ave and Rural Road, day/night) are deliberately NOT part of COMPARISONS
below and are not run by this script -- they are kept, untouched, as
ORIGINAL_COMPARISONS purely for reference. Every already-committed result
for those 4 comparisons, for every model, stays exactly as it was.

Because the new comparisons mix old scenarios (1, 2, 3) with new ones (7,
32, 33), they require position-normalization constants pooled across all 7
scenarios -- otherwise a given normalized position value would not mean the
same physical distance across old and new scenarios. This is why
load_global_normalization() below reads data/global_normalization_all7.json,
a separate file computed for this purpose; the original
data/global_normalization.json (Scenarios 1-4 only) is untouched, so
original_project/ is completely unaffected by this script.

Must be run from the repository root (digital_twin_extended/), e.g.:
    python model_comparison_study/train_model_cross_scenario.py --model mlp
"""
import argparse
import datetime
import json
import os

import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader

from data_feed import DataFeed
from model import MODEL_REGISTRY
from train_model import train_model

DATA_DIR = "data"
RESULT_DIR = "result"

# The original four single-factor comparisons from the methodology (source,
# target, label). Kept here for reference only -- NOT run by default, and
# not touched by this extension. Every already-committed result for these
# 4 comparisons, for every model, stays exactly as it was; see
# ORIGINAL_COMPARISONS below if you ever want to rerun them deliberately.
ORIGINAL_COMPARISONS = [
    ("scenario1", "scenario2", "time_of_day_at_mcallister"),  # McAllister: day -> night
    ("scenario3", "scenario4", "time_of_day_at_ruralroad"),   # Rural Road: day -> night
    ("scenario1", "scenario3", "site_during_day"),            # Day: McAllister -> Rural Road (narrow -> wide)
    ("scenario2", "scenario4", "site_during_night"),          # Night: McAllister -> Rural Road (narrow -> wide)
]

# Lane-width comparisons: Scenarios 1/2 and 32/33 are both 2-lane (narrow)
# sites; Scenario 3 is Rural Road (wide, 6-lane); Scenario 7 is a separate
# 4-lane (wide) site with no day/night pair collected. This is the set
# actually run by this script and by train_baselines_cross_scenario.py
# (both import COMPARISONS), deliberately kept separate from
# ORIGINAL_COMPARISONS above so a run here never touches Scenarios 1-4's
# existing results.
COMPARISONS = [
    ("scenario32", "scenario33", "time_of_day_at_site32"),        # New narrow site: day -> night
    ("scenario1", "scenario32", "site_narrow_to_narrow_day"),     # McAllister -> new narrow site, day
    ("scenario2", "scenario33", "site_narrow_to_narrow_night"),   # McAllister -> new narrow site, night
    ("scenario1", "scenario7", "site_narrow_to_wide_mcallister"), # McAllister (narrow) -> Scenario 7 (wide)
    ("scenario32", "scenario7", "site_narrow_to_wide_site32"),    # New narrow site -> Scenario 7 (wide), replication
    ("scenario3", "scenario7", "site_wide_to_wide"),              # Rural Road (wide) -> Scenario 7 (wide), control
]

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
    # Pooled across all 7 scenarios (1-4, 7, 32, 33) -- see the module
    # docstring. Deliberately a separate file from data/global_normalization.json
    # (Scenarios 1-4 only), which original_project/ still uses unmodified.
    with open(os.path.join(DATA_DIR, "global_normalization_all7.json")) as f:
        norm = json.load(f)
    return norm["max_xy"], norm["max_dist"]


def paths_for(scenario):
    pos_path = os.path.join(DATA_DIR, f"{scenario}_ue_relative_pos.mat")
    pwr_path = os.path.join(DATA_DIR, f"{scenario}_real_beam_pwr.mat")
    return pos_path, pwr_path


def run_comparison(source, target, label, max_xy, max_dist, model_name='mlp'):
    print(f"\n{'=' * 70}\nComparison [{model_name}]: {source} -> {target}  ({label})\n{'=' * 70}", flush=True)

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
            model_name=model_name,
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
    # Always suffixed with the model name (including "mlp") so a comparison
    # rerun never overwrites the original committed stage1_<label>_acc.mat
    # files that the README's reported numbers came from.
    out_acc_path = os.path.join(RESULT_DIR, f"stage1_{label}_{model_name}_acc.mat")
    out_pwr_path = os.path.join(RESULT_DIR, f"stage1_{label}_{model_name}_pwr.mat")
    savemat(out_acc_path, {"acc": all_acc, "sweep_points": [0] + SWEEP_POINTS})
    savemat(out_pwr_path, {"pwr": all_pwr, "sweep_points": [0] + SWEEP_POINTS})
    print(f"Saved {out_acc_path} and {out_pwr_path}", flush=True)

    return all_acc, all_pwr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="mlp",
                         help="Which architecture to run the Stage 1 protocol with.")
    args = parser.parse_args()

    torch.manual_seed(2022)
    os.makedirs(RESULT_DIR, exist_ok=True)
    max_xy, max_dist = load_global_normalization()
    print(f"Using global normalization: max_xy={max_xy:.4f}, max_dist={max_dist:.4f}", flush=True)
    print(f"Running Stage 1 with model={args.model}", flush=True)

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
    print(f"\n=== Stage 1 summary [{args.model}]: mean top-2 accuracy per comparison ===", flush=True)
    header = "real_samples".rjust(12) + "".join(f"{lbl:>28}" for lbl in summary)
    print(header, flush=True)
    for i, p in enumerate(points):
        row = str(p).rjust(12) + "".join(f"{summary[lbl]['top2_accuracy'][i] * 100:>27.2f}%" for lbl in summary)
        print(row, flush=True)

    with open(os.path.join(RESULT_DIR, f"stage1_summary_{args.model}.json"), "w") as f:
        json.dump({"model": args.model, "sweep_points": points, "comparisons": summary}, f, indent=2)

    print("\nStage 1 complete.", flush=True)
