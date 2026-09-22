"""Flag trials that do not belong, before they are averaged into a result.

Every trial is compared against the other trials of the *same* mouse, anesthesia depth,
frequency and metric — never against the whole dataset, because mice legitimately differ
from each other and pooling them would flag whole animals instead of bad trials.

The statistic is a robust z-score:

    z = 0.6745 * (value - median) / MAD

MAD (median absolute deviation) is used instead of SD because a single wild trial inflates
the SD enough to hide itself. The 0.6745 factor puts MAD on the same scale as SD for
normally distributed data, so the usual intuition about z-scores still applies.

Verdicts, per trial and metric:

* `exclude`  |z| > 5 on the response window mean, or the curve is mostly missing
* `inspect`  |z| > 3.5, or the baseline itself is unstable (the animal was not at rest)
* `ok`       everything else

Nothing is removed automatically. The output is a list to look at, because the decision to
drop a trial is a scientific one, not an arithmetic one (issue #20).

Usage:
    python screen_outliers.py --features results/features_per_trial.csv \\
        --curves curves_per_trial.csv --output-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXCLUDE_Z = 5.0
INSPECT_Z = 3.5
# How much of a metric's overall spread must a cell show before its own spread is trusted
# as the yardstick. See robust_z and metric_floor.
SPREAD_FLOOR_FRACTION = 0.75
BASELINE_INSTABILITY_Z = 4.0
MIN_COVERAGE = 0.80


def robust_z(values: pd.Series, floor: float = 0.0) -> pd.Series:
    """Robust z within one cell, with a floor under the spread.

    The floor matters more than it sounds. A cell holds five to nine trials; if six of them
    sit at almost the same value the MAD is near zero and the seventh scores a z of several
    hundred while being physiologically unremarkable. BSR does this constantly, because in
    light anesthesia most trials are exactly 0. The floor is a fraction of the typical
    spread for that metric across all cells, so 'unusual' is judged against how much the
    measurement normally moves, not against how quiet this particular cell happened to be.
    """
    median = values.median()
    mad = (values - median).abs().median()
    scale = max(mad, floor) if np.isfinite(mad) else floor
    if scale <= 0:
        return pd.Series(0.0, index=values.index)
    return 0.6745 * (values - median) / scale


def metric_floor(values: pd.Series) -> float:
    """A metric-wide spread to use as the minimum yardstick inside any one cell.

    Deliberately the ordinary standard deviation rather than a robust one: for BSR most
    trials are exactly zero, so every robust estimate of spread is also zero and the floor
    would not exist where it is needed most.
    """
    spread = float(values.std(ddof=1))
    return SPREAD_FLOOR_FRACTION * spread if np.isfinite(spread) else 0.0


def baseline_stability(curves: pd.DataFrame) -> pd.DataFrame:
    """How much the pre-stimulus period wandered, per trial.

    A trial whose baseline is already drifting cannot have a clean response measured
    against it. The statistic is the spread of the baseline relative to its own cell.
    """
    baseline = curves[curves["time_from_stim"] < 0]
    spread = (
        baseline.groupby(["mouse", "anes", "nominal_frequency_hz", "metric",
                          "recording", "trial_id"])["value"]
        .agg(baseline_spread=lambda s: s.quantile(0.9) - s.quantile(0.1))
        .reset_index()
    )
    floors = spread.groupby("metric")["baseline_spread"].transform(metric_floor)
    spread["_floor"] = floors
    spread["baseline_z"] = (
        spread.groupby(["mouse", "anes", "nominal_frequency_hz", "metric"], group_keys=False)
        .apply(lambda g: robust_z(g["baseline_spread"], floor=float(g["_floor"].iloc[0])))
    )
    return spread.drop(columns="_floor")


def coverage(curves: pd.DataFrame) -> pd.DataFrame:
    """Fraction of the -10..60 s grid that actually has a finite value."""
    expected = len(np.arange(-10, 60.0001, 0.1))
    return (
        curves.groupby(["mouse", "anes", "nominal_frequency_hz", "metric",
                        "recording", "trial_id"])["value"]
        .agg(coverage=lambda s: float(np.isfinite(s).sum()) / expected)
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("results/features_per_trial.csv"))
    parser.add_argument("--curves", type=Path, default=Path("curves_per_trial.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    curves = pd.read_csv(args.curves)

    keys = ["mouse", "anes", "nominal_frequency_hz", "metric"]

    for column in ["mean_0_10", "peak", "auc_0_30"]:
        # One floor per metric, from that metric's overall spread across every trial.
        floors = features.groupby("metric")[column].transform(metric_floor)

        features[f"z_{column}"] = np.nan
        for _, block in features.groupby(keys):
            floor_value = float(floors.loc[block.index].iloc[0]) if len(block) else 0.0
            features.loc[block.index, f"z_{column}"] = robust_z(block[column], floor=floor_value)

    screen = features.merge(baseline_stability(curves),
                            on=keys + ["recording", "trial_id"], how="left")
    screen = screen.merge(coverage(curves), on=keys + ["recording", "trial_id"], how="left")

    z_columns = ["z_mean_0_10", "z_peak", "z_auc_0_30"]
    screen["max_abs_z"] = screen[z_columns].abs().max(axis=1)

    def verdict(row: pd.Series) -> str:
        if row["coverage"] < MIN_COVERAGE:
            return "exclude"
        if row["max_abs_z"] > EXCLUDE_Z:
            return "exclude"
        if row["max_abs_z"] > INSPECT_Z:
            return "inspect"
        if abs(row.get("baseline_z", 0) or 0) > BASELINE_INSTABILITY_Z:
            return "inspect"
        return "ok"

    def reason(row: pd.Series) -> str:
        reasons = []
        if row["coverage"] < MIN_COVERAGE:
            reasons.append(f"only {row['coverage']:.0%} of the trial has data")
        if row["max_abs_z"] > INSPECT_Z:
            worst = max(z_columns, key=lambda c: abs(row[c]))
            reasons.append(f"{worst.removeprefix('z_')} is {row[worst]:+.1f} robust SD "
                           f"from this cell's median")
        if abs(row.get("baseline_z", 0) or 0) > BASELINE_INSTABILITY_Z:
            reasons.append(f"pre-stimulus baseline unusually unstable "
                           f"({row['baseline_z']:+.1f} robust SD)")
        return "; ".join(reasons)

    screen["verdict"] = screen.apply(verdict, axis=1)
    screen["reason"] = screen.apply(reason, axis=1)

    columns = keys + ["recording", "trial_id", "verdict", "reason", "max_abs_z",
                      "baseline_z", "coverage", "mean_0_10", "peak", "auc_0_30"]
    screen = screen[columns].sort_values(["verdict", "max_abs_z"], ascending=[True, False])

    screen.to_csv(args.output_dir / "outlier_screen.csv", index=False)

    counts = screen["verdict"].value_counts()
    print(f"Screened {len(screen):,} trial-metric rows: "
          f"{counts.get('ok', 0)} ok, {counts.get('inspect', 0)} to inspect, "
          f"{counts.get('exclude', 0)} to exclude")

    flagged = screen[screen["verdict"] != "ok"]
    if not flagged.empty:
        print("\nFlagged:")
        for _, row in flagged.iterrows():
            print(f"  [{row['verdict']}] {row['mouse']} {row['anes']} "
                  f"{int(row['nominal_frequency_hz'])}Hz {row['metric']} {row['trial_id']}: "
                  f"{row['reason']}")

    # Which trials are flagged on several metrics at once? Those are the suspect trials,
    # as opposed to a single sensor misbehaving.
    across = (
        flagged.groupby(["mouse", "anes", "recording", "trial_id"])["metric"]
        .agg(list).reset_index()
    )
    across["n_metrics"] = across["metric"].apply(len)
    multi = across[across["n_metrics"] >= 3].sort_values("n_metrics", ascending=False)
    if not multi.empty:
        print("\nTrials flagged on 3+ metrics (likely the whole trial, not one sensor):")
        for _, row in multi.iterrows():
            print(f"  {row['mouse']} {row['anes']} {row['trial_id']} "
                  f"({row['n_metrics']}): {', '.join(sorted(row['metric']))}")


if __name__ == "__main__":
    main()
