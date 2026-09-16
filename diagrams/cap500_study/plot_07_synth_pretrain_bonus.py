"""
BONUS diagram (not one of the three folders you named, but found sitting
alongside them and directly relevant): result/synth_real_s32_rehearsal_500cap_small/
holds a single, standalone experiment that is unlike anything else in this
project's Stage 1-5 -- it is the only place past Stage 0 where a digital
twin is used again.

Protocol (from the result's own summary JSON):
"full_scenario1_synthetic_pretrain_then_250_synthetic_plus_250_real_scenario1_plus_N_scenario32"
i.e.: pretrain on Scenario 1's FULL synthetic digital-twin data (same twin
Stage 0 uses), then rehearse on a fixed 250 synthetic + 250 real Scenario 1
samples UNION N real Scenario 32 (raw, curved) samples, N swept 0->200.

Why this is worth a diagram: every other post-Stage-0 result in this thesis
deliberately avoids the digital twin, using a real-source-trained model as
a stand-in for "already deployed" (see sec:integration's justification).
This experiment is the one place that actually tests the literal claim in
the thesis title -- "digital twin-pretrained beam prediction... under
environmental shift" -- by carrying the twin itself forward into a site
shift, rather than substituting a real-trained proxy for it.

Note: top2_relative_power is NaN at every sweep point for this result
(Scenario 32 here is the RAW, curved version, which -- consistent with
every other NaN case documented in this project -- has a small number of
raw beam-power cells that poison the averaged power ratio); only accuracy
is plotted.

Output (written to result/diagrams/cap500_study/):
    synth_pretrain_scenario32_bonus.png

Run from the repository root:
    python diagrams/cap500_study/plot_07_synth_pretrain_bonus.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from common import INK_PRIMARY, INK_SECONDARY, INK_MUTED, SURFACE, RANDOM_CHANCE_TOP2, style_axes, ensure_out_dir

RESULT_PATH = os.path.join(
    "result", "synth_real_s32_rehearsal_500cap_small",
    "stage2_synth_real_s32_rehearsal_500cap_summary.json",
)


def plot():
    with open(RESULT_PATH) as f:
        d = json.load(f)

    points = d["sweep_points"]
    acc = [v * 100 for v in d["top2_accuracy"]]

    fig, ax = plt.subplots(figsize=(8.5, 6.5), facecolor=SURFACE)
    style_axes(ax)
    ax.axhline(RANDOM_CHANCE_TOP2, color=INK_MUTED, linestyle=":", linewidth=1.2,
               label="Random chance (top-2 of 16 beams)")
    ax.plot(points, acc, color="#7e2f8e", marker="o", linewidth=2.2, markersize=7,
             markeredgecolor=SURFACE, markeredgewidth=1,
             label="Synthetic-pretrained + rehearsed on Scenario 32")

    for x, y in zip(points, acc):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8.5, color=INK_SECONDARY)

    ax.set_xlabel("Real Scenario 32 samples (N)", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylabel("Top-2 accuracy (%)", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylim(0, 100)
    ax.set_title("Bonus: digital-twin-pretrained recovery at a new site\n"
                 "(Scenario 1 synthetic twin -> rehearsed on Scenario 32, raw)",
                 fontsize=12.5, color=INK_PRIMARY, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    out_path = os.path.join(ensure_out_dir(), "synth_pretrain_scenario32_bonus.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
