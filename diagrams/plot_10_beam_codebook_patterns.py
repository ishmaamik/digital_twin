"""
Reproduces the remaining two of the six .fig files in "final figures/":
codebooks.fig (the idealized/theoretical 16-beam steering codebook) and
measured_codebook.fig (the same 16 beams, but using the real, calibrated
antenna beam patterns instead of the idealized array-response formula).
Ported from maltab/plot_beam_codebooks.m, which this script reproduces
numerically rather than visually approximates -- same array-steering math,
same every-4th-beam subsampling used everywhere else in this thesis to go
from the raw 64-beam codebook to the 16-beam one.

Why this is here: these two polar plots show what the "16 beams" that every
top-2/top-5 accuracy number in this thesis is chosen from actually look
like, physically -- narrow, overlapping directional lobes sweeping across
roughly 180 degrees in front of the array. It is setup/hardware context
rather than a result, but it is the only place in this thesis project where
the beam codebook itself -- as opposed to which index was predicted -- is
shown at all.

Data: codebook_beams/beam_angles.mat (the 64 nominal steering angles) and
codebook_beams/built_in_beam_pattern_{0..63}.mat (the real, calibrated
per-beam radiation pattern measured for the physical array) -- confirmed
intact, 66 files present.

Output (written to result/diagrams/):
    beam_codebook_patterns.png -- two polar plots side by side

Run from the repository root:
    python diagrams/plot_10_beam_codebook_patterns.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(__file__))
from common import INK_PRIMARY, SURFACE, ensure_out_dir

CODEBOOK_DIR = "codebook_beams"
NUM_TX = 16       # antenna elements
D_SPACING = 0.5    # element spacing, in wavelengths


def uniform_codebook_power():
    """Idealized steering-vector codebook, evaluated at 0.1-degree angular
    resolution, exactly matching maltab/plot_beam_codebooks.m's first half."""
    beam_angles = loadmat(os.path.join(CODEBOOK_DIR, "beam_angles.mat"))["beam_anlges"].ravel()
    beam_angle_deg = beam_angles / np.pi * 180.0
    # Re-derive a clean, evenly-spaced 64-point angular grid spanning the
    # 2nd through 63rd raw angle (the two endpoints are degenerate), exactly
    # as the original MATLAB script does.
    beam_angle_deg = np.linspace(beam_angle_deg[1], beam_angle_deg[62], 64)

    k_x = np.arange(NUM_TX)
    # Steering vector per candidate beam angle -> (64 angles, 16 elements)
    steering = np.exp(1j * 2 * np.pi * k_x[None, :] * D_SPACING * np.cos(np.deg2rad(beam_angle_deg))[:, None])
    codebook = steering.T  # (16 elements, 64 beams)
    codebook = codebook[:, 1::4]  # every 4th beam, 0-indexed offset 1 -> 16 beams

    test_angles_deg = np.arange(0, 180.1, 0.1)
    array_response = np.exp(-1j * 2 * np.pi * D_SPACING * k_x[None, :] * np.cos(np.deg2rad(test_angles_deg))[:, None])
    # power[beam, angle] = |array_response(angle) . codebook[:, beam]|^2
    power = np.abs(array_response @ codebook) ** 2  # (n_angles, 16 beams)
    power = power.T  # (16 beams, n_angles)
    power = power / power.max()
    return np.deg2rad(test_angles_deg), power


def measured_codebook_power():
    """Real, calibrated per-beam radiation patterns, exactly matching
    maltab/plot_beam_codebooks.m's second half."""
    beam_set = np.arange(64)
    num_beams = len(beam_set)
    sample_pattern = loadmat(os.path.join(CODEBOOK_DIR, "built_in_beam_pattern_0.mat"))["beam_pattern"]
    n_angle_samples = sample_pattern.shape[1]

    pattern = np.zeros((num_beams, n_angle_samples))
    for ii in range(1, num_beams + 1):
        beam_idx = beam_set[num_beams - ii]  # reversed order, matching MATLAB's num_of_beam-ii+1 (1-indexed)
        data = loadmat(os.path.join(CODEBOOK_DIR, f"built_in_beam_pattern_{beam_idx}.mat"))
        row = data["beam_pattern"].ravel().copy()
        row[0] = row[1]  # first sample is always unstable, per original script
        pattern[ii - 1] = row

    pattern = pattern[1::4]  # every 4th row starting at (1-indexed) row 2 -> 16 beams
    pattern = pattern / pattern.max()

    offset = np.deg2rad(4)
    angle_start = np.pi - offset
    angle_end = 0 - offset
    theta = np.linspace(angle_start, angle_end, n_angle_samples)
    return theta, pattern


def plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.5), subplot_kw={"projection": "polar"}, facecolor=SURFACE)

    theta1, power1 = uniform_codebook_power()
    for row in power1:
        ax1.plot(theta1, row, linestyle=":", linewidth=1.1)
    ax1.set_thetamin(0)
    ax1.set_thetamax(180)
    ax1.set_title("Uniform (idealized) beam codebook", fontsize=11.5, color=INK_PRIMARY, pad=18)

    theta2, power2 = measured_codebook_power()
    for row in power2:
        ax2.plot(theta2, row, linewidth=1.1)
    ax2.set_thetamin(0)
    ax2.set_thetamax(180)
    ax2.set_title("Measured (calibrated) beam codebook", fontsize=11.5, color=INK_PRIMARY, pad=18)

    fig.suptitle("The 16-beam steering codebook every top-$k$ result in this thesis is drawn from",
                 fontsize=12.5, color=INK_PRIMARY, y=1.02)
    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "beam_codebook_patterns.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
