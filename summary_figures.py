"""The comparison figures: every condition against every other, on one page.

Three kinds of figure, each answering a different question:

1. `overview_grid.png` — all metrics as rows, anesthesia depth as columns, both frequencies
   drawn together. The figure for "what happened, overall".
2. `bars_<metric>.png` — the mean response over 0-10 s for the four conditions, with SEM
   whiskers and one dot per mouse on top of each bar. Dots matter: with n=4 a bar chart
   without the underlying points hides whether the effect is consistent or driven by one
   animal.
3. `primary_pupil.png` — the pre-registered primary outcome on its own, at full size.

Usage:
    python summary_figures.py --curves curves_per_trial.csv --results-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aggregate import FREQUENCY_COLOURS, axis_label, metric_title


METRIC_ORDER = [
    "pupil_diameter_mm",
    "respiration_resp_bpm",
    "heart_rate_hr_bpm",
    "hrv_sdnn_ms",
    "rr_variability_ms",
    "eeg_rel_delta",
    "eeg_bsr",
]

ANES_ORDER = ["light", "deep"]
ANES_LABEL = {"light": "Light anesthesia", "deep": "Deep anesthesia"}


def overview_grid(grouped: pd.DataFrame, mouse_curves: pd.DataFrame, output: Path) -> None:
    metrics = [m for m in METRIC_ORDER if m in set(grouped["metric"])]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(13, 3.0 * len(metrics)), squeeze=False)

    for row, metric in enumerate(metrics):
        for column, anes in enumerate(ANES_ORDER):
            ax = axes[row][column]
            block = grouped[(grouped["metric"] == metric) & (grouped["anes"] == anes)]

            for frequency, curve in block.groupby("nominal_frequency_hz"):
                curve = curve.sort_values("time_from_stim")
                colour = FREQUENCY_COLOURS.get(int(frequency))
                ax.plot(curve["time_from_stim"], curve["group_mean"], lw=2.2, color=colour,
                        label=f"{int(frequency)} Hz (n={int(curve['n_mice'].max())})")
                if int(curve["n_mice"].max()) > 1:
                    ax.fill_between(curve["time_from_stim"],
                                    curve["group_mean"] - curve["group_sem"],
                                    curve["group_mean"] + curve["group_sem"],
                                    alpha=0.2, color=colour, linewidth=0)

            ax.axhline(0, lw=0.7, color="k", alpha=0.5)
            ax.axvline(0, lw=0.7, color="k", alpha=0.5)
            ax.set_xlim(-10, 60)
            ax.grid(alpha=0.1)
            if row == 0:
                ax.set_title(ANES_LABEL[anes], fontsize=13)
            if column == 0:
                ax.set_ylabel(f"{metric_title(metric)}\n{axis_label(metric)}", fontsize=9)
            if row == len(metrics) - 1:
                ax.set_xlabel("Time from stimulus onset (s)")
            ax.legend(frameon=False, fontsize=8)

    fig.suptitle("All metrics, both anesthesia depths — group mean ± SEM", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def bar_figure(features: pd.DataFrame, metric: str, output: Path, feature: str = "mean_0_10") -> None:
    """Group mean with SEM and one dot per mouse, for the four conditions."""
    block = features[features["metric"] == metric]
    per_mouse = (
        block.groupby(["mouse", "anes", "nominal_frequency_hz"])[feature]
        .median()
        .reset_index()
    )

    conditions = [(anes, frequency) for anes in ANES_ORDER for frequency in (40, 100)]
    positions = [0, 0.85, 2.2, 3.05]

    fig, ax = plt.subplots(figsize=(7.5, 5))

    for position, (anes, frequency) in zip(positions, conditions):
        values = per_mouse[(per_mouse["anes"] == anes)
                           & (per_mouse["nominal_frequency_hz"] == frequency)][feature]
        if values.empty:
            continue

        mean = values.mean()
        sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
        colour = FREQUENCY_COLOURS[frequency]

        ax.bar(position, mean, width=0.75, color=colour, alpha=0.75,
               label=f"{frequency} Hz" if anes == "light" else None)
        if np.isfinite(sem):
            ax.errorbar(position, mean, yerr=sem, color="k", capsize=5, lw=1.4)

        jitter = np.linspace(-0.18, 0.18, len(values))
        ax.scatter(position + jitter, values, color="k", s=26, zorder=3, alpha=0.8)
        ax.text(position, ax.get_ylim()[1] * 0.02, f"n={len(values)}",
                ha="center", fontsize=8, color="#444")

    ax.axhline(0, lw=0.8, color="k")
    ax.set_xticks([np.mean(positions[:2]), np.mean(positions[2:])])
    ax.set_xticklabels([ANES_LABEL[a] for a in ANES_ORDER])
    ax.set_ylabel(f"{axis_label(metric)}, mean over 0-10 s")
    ax.set_title(f"{metric_title(metric)} — response by condition\n"
                 f"bars are group mean ± SEM, dots are individual mice", fontsize=11)
    ax.legend(frameon=False)
    ax.grid(alpha=0.12, axis="y")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def primary_figure(grouped: pd.DataFrame, mouse_curves: pd.DataFrame, output: Path) -> None:
    """The pupil, both depths side by side, individual mice visible."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, anes in zip(axes, ANES_ORDER):
        block = grouped[(grouped["metric"] == "pupil_diameter_mm") & (grouped["anes"] == anes)]
        individual = mouse_curves[(mouse_curves["metric"] == "pupil_diameter_mm")
                                  & (mouse_curves["anes"] == anes)]

        for frequency, curve in block.groupby("nominal_frequency_hz"):
            curve = curve.sort_values("time_from_stim")
            colour = FREQUENCY_COLOURS[int(frequency)]

            for _, mouse_curve in individual[
                    individual["nominal_frequency_hz"] == frequency].groupby("mouse"):
                mouse_curve = mouse_curve.sort_values("time_from_stim")
                ax.plot(mouse_curve["time_from_stim"], mouse_curve["mouse_median"],
                        lw=0.9, alpha=0.3, color=colour)

            ax.plot(curve["time_from_stim"], curve["group_mean"], lw=3, color=colour,
                    label=f"{int(frequency)} Hz (n={int(curve['n_mice'].max())} mice)")
            if int(curve["n_mice"].max()) > 1:
                ax.fill_between(curve["time_from_stim"],
                                curve["group_mean"] - curve["group_sem"],
                                curve["group_mean"] + curve["group_sem"],
                                alpha=0.22, color=colour, linewidth=0)

        ax.axhline(0, lw=0.8, color="k", alpha=0.6)
        ax.axvline(0, lw=0.8, color="k", alpha=0.6)
        ax.set_xlim(-10, 60)
        ax.set_title(ANES_LABEL[anes], fontsize=13)
        ax.set_xlabel("Time from stimulus onset (s)")
        ax.grid(alpha=0.12)
        ax.legend(frameon=False)

    axes[0].set_ylabel("Pupil diameter, change from baseline (%)")
    fig.suptitle("Pupil dilation in response to nociceptive stimulation", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    grouped = pd.read_csv(args.results_dir / "curves_group.csv")
    mouse_curves = pd.read_csv(args.results_dir / "curves_per_mouse.csv")
    features = pd.read_csv(args.results_dir / "features_per_trial.csv")

    summary = args.results_dir / "summary_figures"
    summary.mkdir(parents=True, exist_ok=True)

    overview_grid(grouped, mouse_curves, summary / "overview_grid.png")
    primary_figure(grouped, mouse_curves, summary / "primary_pupil.png")

    for metric in METRIC_ORDER:
        if metric in set(features["metric"]):
            bar_figure(features, metric, summary / f"bars_{metric}.png")

    print(f"Summary figures in {summary}")


if __name__ == "__main__":
    main()
