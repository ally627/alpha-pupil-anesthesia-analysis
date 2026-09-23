"""Figure 1: pupil response, time course and quantification in one figure.

Panels A and B show the time course under each anesthesia depth; panel C
quantifies the same trials as the mean over 0-10 s. Keeping the three panels
in one figure avoids presenting the curves and their summary as two separate
figures, which would read as showing the same data twice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC = "pupil_diameter_mm"
ANES_ORDER = ("light", "deep")
ANES_LABEL = {"light": "Light anesthesia", "deep": "Deep anesthesia"}
FREQUENCY_COLOURS = {40: "#d1495b", 100: "#2e86ab"}
PLOT_START_S = -10.0
PLOT_END_S = 60.0


def build(results_dir: Path, output: Path) -> None:
    group = pd.read_csv(results_dir / "curves_group.csv")
    per_mouse = pd.read_csv(results_dir / "curves_per_mouse.csv")
    features = pd.read_csv(results_dir / "features_per_mouse.csv")

    group = group[group["metric"] == METRIC]
    per_mouse = per_mouse[per_mouse["metric"] == METRIC]
    features = features[features["metric"] == METRIC]

    fig = plt.figure(figsize=(16.5, 5.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.72], wspace=0.28)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]

    # --- panels A and B: time course -------------------------------------
    for ax, anes in zip(axes[:2], ANES_ORDER):
        for frequency, colour in FREQUENCY_COLOURS.items():
            block = group[(group["anes"] == anes)
                          & (group["nominal_frequency_hz"] == frequency)]
            if block.empty:
                continue
            block = block[(block["time_from_stim"] >= PLOT_START_S)
                          & (block["time_from_stim"] <= PLOT_END_S)]

            mice = per_mouse[(per_mouse["anes"] == anes)
                             & (per_mouse["nominal_frequency_hz"] == frequency)]
            for _, trace in mice.groupby("mouse"):
                trace = trace[(trace["time_from_stim"] >= PLOT_START_S)
                              & (trace["time_from_stim"] <= PLOT_END_S)]
                ax.plot(trace["time_from_stim"], trace["mouse_median"],
                        color=colour, lw=0.7, alpha=0.32)

            n_mice = int(block["n_mice"].max())
            n_trials = int(block["n_trials"].max())
            ax.plot(block["time_from_stim"], block["group_mean"], color=colour, lw=2.4,
                    label=f"{frequency} Hz (n={n_mice} mice, {n_trials} trials)")
            ax.fill_between(block["time_from_stim"],
                            block["group_mean"] - block["group_sem"],
                            block["group_mean"] + block["group_sem"],
                            color=colour, alpha=0.18, lw=0)

        ax.axvline(0, color="k", lw=0.9, ls="--", alpha=0.6)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Time from stimulus onset (s)")
        ax.set_title(ANES_LABEL[anes], fontsize=12)
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.12)

    axes[0].set_ylabel("Pupil diameter, change from baseline (%)")
    shared = [axes[0].get_ylim(), axes[1].get_ylim()]
    low = min(limit[0] for limit in shared)
    high = max(limit[1] for limit in shared)
    axes[0].set_ylim(low, high)
    axes[1].set_ylim(low, high)

    # --- panel C: quantification -----------------------------------------
    ax = axes[2]
    conditions = [(anes, frequency) for anes in ANES_ORDER for frequency in (40, 100)]
    positions = [0.0, 0.85, 2.2, 3.05]

    for position, (anes, frequency) in zip(positions, conditions):
        values = features[(features["anes"] == anes)
                          & (features["nominal_frequency_hz"] == frequency)]["mean_0_10"]
        if values.empty:
            continue
        mean = values.mean()
        sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
        ax.bar(position, mean, width=0.75, color=FREQUENCY_COLOURS[frequency], alpha=0.75)
        if np.isfinite(sem):
            ax.errorbar(position, mean, yerr=sem, color="k", capsize=5, lw=1.4)
        jitter = np.linspace(-0.18, 0.18, len(values))
        ax.scatter(position + jitter, values, color="k", s=26, zorder=3, alpha=0.8)
        ax.text(position, 1.5, f"n={len(values)}", ha="center", fontsize=8, color="#444")

    ax.axhline(0, lw=0.8, color="k")
    ax.set_xticks(positions)
    ax.set_xticklabels(["40 Hz", "100 Hz", "40 Hz", "100 Hz"], fontsize=9)
    ax.text(np.mean(positions[:2]), -0.14, ANES_LABEL["light"], transform=
            ax.get_xaxis_transform(), ha="center", fontsize=11)
    ax.text(np.mean(positions[2:]), -0.14, ANES_LABEL["deep"], transform=
            ax.get_xaxis_transform(), ha="center", fontsize=11)
    ax.set_ylabel("Pupil diameter, mean over 0-10 s (%)")
    ax.grid(alpha=0.12, axis="y")

    for ax, letter in zip(axes, "ABC"):
        ax.text(-0.09, 1.04, letter, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom")

    fig.suptitle("Pupil dilation in response to nociceptive stimulation", fontsize=15)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/summary_figures/figure1_pupil_combined.png"))
    args = parser.parse_args()
    build(args.results_dir, args.output)


if __name__ == "__main__":
    main()
