"""
Reproduces the original Jiang & Alkhateeb ("Paper A") Figure 5 -- one of the
six .fig files found in "final figures/" (performance_vs_num_data.fig),
whose generating MATLAB script (maltab/plot_performance_fig5.m) this script
ports to Python so it can be embedded directly in the LaTeX thesis as a PNG
(LaTeX cannot include a native .fig file).

What this figure shows: NO transfer learning here -- three independent
"trained from scratch" regimes, each trained on a growing number of data
points (10 to 200) and evaluated on the same real held-out test set:
  1. Trained on REAL data only.
  2. Trained on SYNTHETIC (digital-twin) data only, measured codebook.
  3. Trained on SYNTHETIC (digital-twin) data only, uniform (idealized) codebook.
This is Paper A's own baseline evidence for why pretraining on a twin is
worth doing at all -- it shows synthetic-only training is a reasonable but
not great stand-in for real data, motivating the twin-then-fine-tune recipe
that Figure 6 (see plot_09) then validates.

Data: result/non-rehearsals/all_{acc,pwr}_train_on_{real,synth_measured,synth_uniform}.mat
-- confirmed intact and unmodified from the original project; each array has
shape (4 top-k values, 30 seeds, 20 sweep points from 10 to 200 samples).
Index 1 (0-indexed) = top-2, matching every other top-2-primary figure in
this thesis.

Output (written to result/diagrams/):
    paper_a_fig5_synthetic_vs_real.png

Run from the repository root:
    python diagrams/plot_08_paper_a_synthetic_vs_real.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(__file__))
from common import INK_PRIMARY, INK_SECONDARY, SURFACE, style_axes, ensure_out_dir

RESULT_DIR = os.path.join("result", "non-rehearsals")
NUM_DATA = np.arange(10, 201, 10)  # 20 points, matches the .mat arrays' last axis
TOPK_INDEX = 1  # top-2

CONDITIONS = [
    ("real", "Trained on real data", "#0072BD"),
    ("synth_measured", "Trained on synthetic (measured codebook)", "#A2142F"),
    ("synth_uniform", "Trained on synthetic (uniform codebook)", "#ff13a6"),
]


def load_condition(name):
    acc = loadmat(os.path.join(RESULT_DIR, f"all_acc_train_on_{name}.mat"))[f"all_acc_train_on_{name}"]
    pwr = loadmat(os.path.join(RESULT_DIR, f"all_pwr_train_on_{name}.mat"))[f"all_pwr_train_on_{name}"]
    acc_mean = acc[TOPK_INDEX].mean(axis=0) * 100  # mean over 30 seeds -> (20,)
    pwr_mean = pwr[TOPK_INDEX].mean(axis=0) * 100
    return acc_mean, pwr_mean


def plot():
    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=SURFACE)
    style_axes(ax)

    for name, label, color in CONDITIONS:
        acc_mean, pwr_mean = load_condition(name)
        ax.plot(NUM_DATA, acc_mean, "--s", color=color, linewidth=1.8, markersize=5,
                 label=f"{label} -- accuracy")
        ax.plot(NUM_DATA, pwr_mean, "-o", color=color, linewidth=1.8, markersize=5,
                 label=f"{label} -- relative power")

    ax.set_xlabel("Number of training data points", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylabel("Top-2 accuracy / relative power (%)", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylim(30, 100)
    ax.set_title("Reproduction of Paper A Figure 5: training-from-scratch on\n"
                 "real vs. synthetic (digital-twin) data, no transfer learning",
                 fontsize=12.5, color=INK_PRIMARY, pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "paper_a_fig5_synthetic_vs_real.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
