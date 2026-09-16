"""
Reproduces the original Jiang & Alkhateeb ("Paper A") Figure 6 -- the single
most important of the six .fig files in "final figures/"
(transfer_learning_performance.fig). Its generating MATLAB script
(maltab/plot_performance_transfer_learning_fig6.m) is ported to Python here
so it can be embedded as a PNG in the LaTeX thesis.

What this figure shows: THIS is Paper A's headline result -- the one Stage 0
of this thesis reproduces numerically (91.53% vs. the published 91.4%) but
has never actually been re-plotted as a figure anywhere in this project
until now. Three regimes, each evaluated on the same real held-out test set
as a function of the number of REAL samples used:
  1. Trained on real data only, from scratch (no twin at all).
  2. Transfer learning: pretrained on the synthetic digital twin (measured
     codebook), then fine-tuned on a growing number of real samples.
  3. Transfer learning: pretrained on the synthetic digital twin (uniform
     codebook), then fine-tuned on a growing number of real samples.
The visual signature of Paper A's claim is the transfer-learning curves
starting far above the train-on-real curve at small real-sample counts (the
digital twin gives a "head start" for free) and the gap closing as more real
data becomes available -- exactly the sample-efficiency property this
thesis's Stage 1 onward asks whether a post-deployment environmental shift
erodes.

Data: result/non-rehearsals/all_{acc,pwr}_train_on_real_compare_to_transfer.mat
and all_{acc,pwr}_train_on_transfer_{measured,uniform}.mat -- confirmed
intact. Index 1 (0-indexed) = top-2, matching every other top-2-primary
figure in this thesis. Note the two families of curves are swept at
slightly different sample counts (5-100 for the real-only baseline; 0-100,
including the zero-shot point, for both transfer-learning curves) --
plotted on their own true x-values rather than forced onto a shared axis
that would misrepresent either one.

Output (written to result/diagrams/):
    paper_a_fig6_transfer_learning.png

Run from the repository root:
    python diagrams/plot_09_paper_a_transfer_learning.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(__file__))
from common import INK_PRIMARY, INK_SECONDARY, SURFACE, style_axes, ensure_out_dir

RESULT_DIR = os.path.join("result", "non-rehearsals")
TOPK_INDEX = 1  # top-2


def load(varname):
    acc = loadmat(os.path.join(RESULT_DIR, f"all_acc_train_on_{varname}.mat"))[f"all_acc_train_on_{varname}"]
    pwr = loadmat(os.path.join(RESULT_DIR, f"all_pwr_train_on_{varname}.mat"))[f"all_pwr_train_on_{varname}"]
    return acc[TOPK_INDEX].mean(axis=0) * 100, pwr[TOPK_INDEX].mean(axis=0) * 100


def plot():
    real_x = np.arange(5, 101, 5)     # 20 points
    transfer_x = np.arange(0, 101, 5)  # 21 points, includes the zero-shot point

    real_acc, real_pwr = load("real_compare_to_transfer")
    trans_meas_acc, trans_meas_pwr = load("transfer_measured")
    trans_unif_acc, trans_unif_pwr = load("transfer_uniform")

    fig, ax = plt.subplots(figsize=(9.5, 6.5), facecolor=SURFACE)
    style_axes(ax)

    ax.plot(real_x, real_acc, "--s", color="#0072BD", linewidth=1.8, markersize=5,
             label="Trained on real only -- accuracy")
    ax.plot(real_x, real_pwr, "-o", color="#0072BD", linewidth=1.8, markersize=5,
             label="Trained on real only -- relative power")

    ax.plot(transfer_x, trans_meas_acc, "--s", color="#7e2f8e", linewidth=1.8, markersize=5,
             label="Transfer learning (measured codebook) -- accuracy")
    ax.plot(transfer_x, trans_meas_pwr, "-o", color="#7e2f8e", linewidth=1.8, markersize=5,
             label="Transfer learning (measured codebook) -- relative power")

    ax.plot(transfer_x, trans_unif_acc, "--s", color="#946801", linewidth=1.8, markersize=5,
             label="Transfer learning (uniform codebook) -- accuracy")
    ax.plot(transfer_x, trans_unif_pwr, "-o", color="#946801", linewidth=1.8, markersize=5,
             label="Transfer learning (uniform codebook) -- relative power")

    ax.set_xlabel("Number of real data points used for training", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylabel("Top-2 accuracy / relative power (%)", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylim(25, 100)
    ax.set_title("Reproduction of Paper A Figure 6: digital-twin-pretrained\n"
                 "transfer learning vs. training on real data alone",
                 fontsize=12.5, color=INK_PRIMARY, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "paper_a_fig6_transfer_learning.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
