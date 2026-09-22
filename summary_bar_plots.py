from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from group_plots import load_combined_files


OUTPUT_DIR = Path("summary_bar_plots")
WINDOW_START = 0
WINDOW_END = 10

METRIC_ORDER = [
    "pupil_diameter_mm",
    "respiration_resp_bpm",
    "heart_rate_hr_bpm",
    "eeg_rel_delta",
]

METRIC_LABELS = {
    "pupil_diameter_mm": "Pupil diameter",
    "respiration_resp_bpm": "Respiration rate",
    "heart_rate_hr_bpm": "Heart rate",
    "eeg_rel_delta": "EEG rel_delta",
}

METRIC_COLORS = {
    "pupil_diameter_mm": "#4C78A8",
    "respiration_resp_bpm": "#59A14F",
    "heart_rate_hr_bpm": "#F28E2B",
    "eeg_rel_delta": "#B07AA1",
}


def calculate_summary(data):
    data = data[
        data["metric"].isin(METRIC_ORDER)
        & data["time_from_stim"].between(WINDOW_START, WINDOW_END)
    ].copy()

    # First calculate one value for each mouse in every condition.
    per_mouse = (
        data.groupby(
            ["mouse", "anes", "frequency_hz", "metric"],
            as_index=False,
        )["mean"]
        .mean()
        .rename(columns={"mean": "mean_change_0_10s"})
    )

    # Then calculate the group mean and SD between mice.
    summary = (
        per_mouse.groupby(
            ["anes", "frequency_hz", "metric"],
            as_index=False,
        )
        .agg(
            group_mean=("mean_change_0_10s", "mean"),
            group_sd=("mean_change_0_10s", "std"),
            n_mice=("mouse", "nunique"),
        )
    )

    # group_sd stays NaN for a single-mouse cell. matplotlib simply draws no whisker for
    # NaN, which is the honest rendering; filling it with 0 drew a confident zero-length
    # error bar on the least certain bar in the figure. See issue #6.
    return per_mouse, summary


def calculate_common_y_limits(summary):
    sd = summary["group_sd"].fillna(0)
    lower = (summary["group_mean"] - sd).min()
    upper = (summary["group_mean"] + sd).max()

    lower = min(lower, 0)
    upper = max(upper, 0)

    padding = max((upper - lower) * 0.1, 2)

    return lower - padding, upper + padding


def make_bar_plot(summary, anes, frequency, y_limits):
    selected = summary[
        (summary["anes"] == anes)
        & (summary["frequency_hz"] == frequency)
    ].copy()

    selected = (
        selected.set_index("metric")
        .reindex(METRIC_ORDER)
        .reset_index()
    )

    if selected["group_mean"].isna().any():
        missing = selected.loc[
            selected["group_mean"].isna(), "metric"
        ].tolist()
        print(
            f"Skipping {anes}, {frequency} Hz: "
            f"missing metrics {missing}"
        )
        return

    labels = [
        f"{METRIC_LABELS[m]}\n(n={int(n)})"
        for m, n in zip(selected["metric"], selected["n_mice"])
    ]
    colors = [METRIC_COLORS[m] for m in selected["metric"]]

    plt.figure(figsize=(9, 6))

    plt.bar(
        labels,
        selected["group_mean"],
        yerr=selected["group_sd"],
        color=colors,
        capsize=6,
        edgecolor="black",
        linewidth=0.8,
    )

    plt.axhline(0, color="black", linewidth=0.9)
    plt.ylim(y_limits)

    plt.ylabel("Mean change from baseline, 0–10 s (%)")

    # n is whatever survived quality control in this condition, which is not always 4.
    n_by_metric = dict(zip(selected["metric"], selected["n_mice"].astype(int)))
    n_values = sorted(set(n_by_metric.values()))
    n_text = f"n={n_values[0]} mice" if len(n_values) == 1 else f"n={min(n_values)}–{max(n_values)} mice"

    plt.title(
        f"Physiological responses – {anes} anesthesia – "
        f"{frequency} Hz – {n_text}"
    )

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    output_name = f"summary_{anes}_{frequency}Hz.png"
    plt.savefig(OUTPUT_DIR / output_name, dpi=200)
    plt.close()

    print(f"Saved: {output_name}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = load_combined_files()

    if data.empty:
        print("No combined data found.")
        return

    per_mouse, summary = calculate_summary(data)

    per_mouse.to_csv(
        OUTPUT_DIR / "summary_0_10s_per_mouse.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "summary_0_10s_group.csv",
        index=False,
    )

    y_limits = calculate_common_y_limits(summary)

    for anes in ["light", "deep"]:
        for frequency in [40, 100]:
            make_bar_plot(
                summary=summary,
                anes=anes,
                frequency=frequency,
                y_limits=y_limits,
            )

    print("Done summary bar plots.")


if __name__ == "__main__":
    main()