"""Aggregate per-trial curves into per-mouse and group curves, and plot both.

Aggregation rule, decided with Ally's supervisor (issue #14):

* **Within a mouse, across trials: median.** Trials are technical repeats of the same
  stimulus in the same animal, so a single large excursion is far more likely to be a
  blink, a movement or a tracking failure than a different physiological truth.
* **Across mice: mean, with SEM.** Each mouse is a real biological sample, n is 3-4, and a
  median over three values throws away most of the information.

The group band is the **SEM** (`SD / sqrt(n_mice)`), not the SD: it describes how precisely
the group mean is known, which is what a group figure is claiming (issue #13). The per-mouse
curves are drawn behind it so the spread stays visible rather than only summarised.

Usage:
    python aggregate.py --curves curves_per_trial.csv --output-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from derived_metrics import METRIC_BY_KEY
from quality_control import MIN_TRIALS_PER_CELL


FREQUENCY_COLOURS = {40: "#d1495b", 100: "#2e86ab"}
MOUSE_ORDER_KEY = lambda name: (len(name), name)


def metric_title(key: str) -> str:
    return {
        "pupil_diameter_mm": "Pupil diameter",
        "heart_rate_hr_bpm": "Heart rate",
        "hrv_sdnn_ms": "Heart rate variability (SDNN-10s)",
        "respiration_resp_bpm": "Respiration rate",
        "rr_variability_ms": "Respiratory interval variability",
        "eeg_rel_delta": "EEG relative delta power",
        "eeg_bsr": "Burst-suppression ratio",
    }.get(key, key)


def axis_label(key: str) -> str:
    entry = METRIC_BY_KEY.get(key)
    return entry[4] if entry else "Value"


def per_mouse_curves(curves: pd.DataFrame) -> pd.DataFrame:
    """Median across trials, within mouse / anesthesia / frequency / metric / time."""
    trial_counts = (
        curves.groupby(["mouse", "anes", "nominal_frequency_hz", "metric"])["trial_id"]
        .nunique()
        .rename("n_trials")
        .reset_index()
    )

    curves = curves.merge(trial_counts, on=["mouse", "anes", "nominal_frequency_hz", "metric"])
    kept = curves[curves["n_trials"] >= MIN_TRIALS_PER_CELL]

    dropped = trial_counts[trial_counts["n_trials"] < MIN_TRIALS_PER_CELL]
    for _, row in dropped.iterrows():
        print(f"Excluded cell: {row['mouse']} {row['anes']} {int(row['nominal_frequency_hz'])}Hz "
              f"{row['metric']} — {int(row['n_trials'])} trial(s)")

    return (
        kept.groupby(["mouse", "anes", "nominal_frequency_hz", "metric",
                      "time_from_stim", "n_trials"], as_index=False)["value"]
        .median()
        .rename(columns={"value": "mouse_median"})
    )


def group_curves(mouse_curves: pd.DataFrame) -> pd.DataFrame:
    """Mean across mice, with SD, SEM and n."""
    grouped = (
        mouse_curves.groupby(["anes", "nominal_frequency_hz", "metric", "time_from_stim"])
        .agg(
            group_mean=("mouse_median", "mean"),
            group_sd=("mouse_median", "std"),
            n_mice=("mouse", "nunique"),
            n_trials=("n_trials", "sum"),
        )
        .reset_index()
    )
    grouped["group_sem"] = grouped["group_sd"] / np.sqrt(grouped["n_mice"])
    return grouped


def plot_group(grouped: pd.DataFrame, mouse_curves: pd.DataFrame, output_dir: Path) -> None:
    """One figure per metric and anesthesia depth, both frequencies together."""
    plots = output_dir / "group_plots"
    plots.mkdir(parents=True, exist_ok=True)

    for (metric, anes), block in grouped.groupby(["metric", "anes"]):
        plt.figure(figsize=(10, 5.5))

        for frequency, curve in block.groupby("nominal_frequency_hz"):
            curve = curve.sort_values("time_from_stim")
            colour = FREQUENCY_COLOURS.get(int(frequency), None)
            n_mice = int(curve["n_mice"].max())
            n_trials = int(curve["n_trials"].max())

            # The individual mice behind the mean, thin and pale.
            individual = mouse_curves[
                (mouse_curves["metric"] == metric)
                & (mouse_curves["anes"] == anes)
                & (mouse_curves["nominal_frequency_hz"] == frequency)
            ]
            for _, mouse_curve in individual.groupby("mouse"):
                mouse_curve = mouse_curve.sort_values("time_from_stim")
                plt.plot(mouse_curve["time_from_stim"], mouse_curve["mouse_median"],
                         lw=0.9, alpha=0.35, color=colour)

            plt.plot(curve["time_from_stim"], curve["group_mean"], lw=2.8, color=colour,
                     label=f"{int(frequency)} Hz  (n={n_mice} mice, {n_trials} trials)")

            if n_mice > 1:
                plt.fill_between(
                    curve["time_from_stim"],
                    curve["group_mean"] - curve["group_sem"],
                    curve["group_mean"] + curve["group_sem"],
                    alpha=0.22, color=colour, linewidth=0,
                )

        plt.axhline(0, lw=0.8, color="k", alpha=0.6)
        plt.axvline(0, lw=0.8, color="k", alpha=0.6)
        plt.xlim(-10, 60)
        plt.xlabel("Time from stimulus onset (s)")
        plt.ylabel(axis_label(metric))
        plt.title(f"{metric_title(metric)} — {anes} anesthesia\n"
                  f"group mean ± SEM, thin lines are individual mice")
        plt.legend(frameon=False)
        plt.grid(alpha=0.12)
        plt.tight_layout()
        plt.savefig(plots / f"group_{metric}_{anes}.png", dpi=200)
        plt.close()

    print(f"Group plots in {plots}")


def plot_per_mouse(mouse_curves: pd.DataFrame, output_dir: Path) -> None:
    """One panel per metric and anesthesia depth, a subplot per mouse, shared y axis.

    This is the screening figure: anything odd in one animal is visible before it is
    averaged away. See issue #20.
    """
    plots = output_dir / "per_mouse_plots"
    plots.mkdir(parents=True, exist_ok=True)

    for (metric, anes), block in mouse_curves.groupby(["metric", "anes"]):
        mice = sorted(block["mouse"].unique(), key=MOUSE_ORDER_KEY)
        fig, axes = plt.subplots(1, len(mice), figsize=(4.2 * len(mice), 4.4),
                                 sharey=True, squeeze=False)

        for ax, mouse in zip(axes[0], mice):
            per_mouse = block[block["mouse"] == mouse]

            for frequency, curve in per_mouse.groupby("nominal_frequency_hz"):
                curve = curve.sort_values("time_from_stim")
                ax.plot(curve["time_from_stim"], curve["mouse_median"], lw=2.2,
                        color=FREQUENCY_COLOURS.get(int(frequency)),
                        label=f"{int(frequency)} Hz (n={int(curve['n_trials'].iloc[0])})")

            ax.axhline(0, lw=0.8, color="k", alpha=0.6)
            ax.axvline(0, lw=0.8, color="k", alpha=0.6)
            ax.set_xlim(-10, 60)
            ax.set_title(mouse)
            ax.set_xlabel("Time from stimulus (s)")
            ax.grid(alpha=0.12)
            ax.legend(frameon=False, fontsize=8)

        axes[0][0].set_ylabel(axis_label(metric))
        fig.suptitle(f"{metric_title(metric)} — {anes} anesthesia — per mouse "
                     f"(median across that mouse's trials)")
        fig.tight_layout()
        fig.savefig(plots / f"per_mouse_{metric}_{anes}.png", dpi=170)
        plt.close(fig)

    print(f"Per-mouse plots in {plots}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=Path("curves_per_trial.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    curves = pd.read_csv(args.curves)

    mouse_curves = per_mouse_curves(curves)
    grouped = group_curves(mouse_curves)

    mouse_curves.to_csv(args.output_dir / "curves_per_mouse.csv", index=False)
    grouped.to_csv(args.output_dir / "curves_group.csv", index=False)

    plot_per_mouse(mouse_curves, args.output_dir)
    plot_group(grouped, mouse_curves, args.output_dir)

    print(f"\n{mouse_curves['mouse'].nunique()} mice, "
          f"{mouse_curves['metric'].nunique()} metrics, "
          f"{len(grouped):,} group rows")


if __name__ == "__main__":
    main()
