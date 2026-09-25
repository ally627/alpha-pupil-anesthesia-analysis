"""Figure 7: response strength of every metric on one shared scale.

Each metric has its own units, so they cannot be compared directly. For every
trial the mean change over 0-10 s is divided by that metric's own pre-stimulus
standard deviation (-10..0 s; the per-cell median of the trial SDs, so a single
unusually quiet trial does not inflate the ratio). The result is unitless and
signed: negative means the metric moved in the opposite direction.

Aggregation matches the rest of the analysis: median within mouse, mean and
SEM across mice. A mouse contributes to a condition only if it has at least
MIN_TRIALS_PER_CELL trials there - the same rule as aggregate.py and
features.py, so every figure reports the same n.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_TRIALS_PER_CELL = 3
BASELINE_START_S = -10.0
BASELINE_END_S = 0.0
RESPONSE_START_S = 0.0
RESPONSE_END_S = 10.0

METRICS = (
    ("pupil_diameter_mm", "Pupil", "#d1495b"),
    ("respiration_resp_bpm", "Respiration", "#2e86ab"),
    ("heart_rate_hr_bpm", "Heart rate", "#ffa502"),
    ("eeg_rel_delta", "EEG rel. delta", "#7b68ee"),
    ("hrv_sdnn_ms", "HRV (SDNN)", "#2ec4a6"),
)
CONDITIONS = (("light", 40), ("light", 100), ("deep", 40), ("deep", 100))
CONDITION_LABELS = ("Light 40 Hz", "Light 100 Hz", "Deep 40 Hz", "Deep 100 Hz")
CELL = ["mouse", "anes", "nominal_frequency_hz", "metric"]


def per_mouse_strength(curves: pd.DataFrame) -> pd.DataFrame:
    keys = CELL + ["trial_id"]
    time = curves["time_from_stim"]
    baseline = (curves[(time >= BASELINE_START_S) & (time < BASELINE_END_S)]
                .groupby(keys)["value"].std().rename("baseline_sd"))
    response = (curves[(time >= RESPONSE_START_S) & (time <= RESPONSE_END_S)]
                .groupby(keys)["value"].mean().rename("response"))
    trials = pd.concat([baseline, response], axis=1).reset_index()

    trials["n_trials"] = trials.groupby(CELL)["trial_id"].transform("nunique")
    trials = trials[trials["n_trials"] >= MIN_TRIALS_PER_CELL]

    trials["cell_sd"] = trials.groupby(CELL)["baseline_sd"].transform("median")
    trials["strength"] = trials["response"] / trials["cell_sd"]
    return trials.groupby(CELL)["strength"].median().reset_index()


def build(curves_path: Path, output: Path, table_output: Path) -> None:
    curves = pd.read_csv(curves_path)
    per_mouse = per_mouse_strength(curves)

    rows = []
    fig, ax = plt.subplots(figsize=(12, 6.2))
    width = 0.16
    x = np.arange(len(CONDITIONS))

    for index, (metric, label, colour) in enumerate(METRICS):
        block = per_mouse[per_mouse["metric"] == metric]
        means, sems, points = [], [], []
        for anes, frequency in CONDITIONS:
            values = block[(block["anes"] == anes)
                           & (block["nominal_frequency_hz"] == frequency)]["strength"].values
            mean = values.mean() if len(values) else np.nan
            sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
            means.append(mean)
            sems.append(sem)
            points.append(values)
            rows.append({"metric": metric, "anes": anes, "nominal_frequency_hz": frequency,
                         "n_mice": len(values), "mean": mean, "sem": sem})

        positions = x + (index - 2) * width
        ax.bar(positions, means, width=width * 0.92, color=colour, alpha=0.85, label=label)
        ax.errorbar(positions, means, yerr=sems, fmt="none", ecolor="k", capsize=3, lw=1.1)
        for position, values in zip(positions, points):
            jitter = np.linspace(-0.045, 0.045, len(values)) if len(values) else []
            ax.scatter(np.full(len(values), position) + jitter, values,
                       s=13, color="k", zorder=3, alpha=0.75)

    ax.axhline(0, lw=1.0, color="k")
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_LABELS, fontsize=11)
    ax.set_ylabel("Response (multiples of pre-stimulus SD, signed)")
    ax.set_title("Response magnitude across metrics", fontsize=14)
    ax.legend(frameon=False, ncol=5, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(alpha=0.12, axis="y")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(table_output, index=False)
    print(f"wrote {output} and {table_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=Path("curves_per_trial.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/summary_figures/cross_metric_strength_signed.png"))
    parser.add_argument("--table", type=Path, default=Path("results/cross_metric_strength.csv"))
    args = parser.parse_args()
    build(args.curves, args.output, args.table)


if __name__ == "__main__":
    main()
