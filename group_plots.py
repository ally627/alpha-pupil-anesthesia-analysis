from pathlib import Path
import re

import pandas as pd

from quality_control import MIN_TRIALS_PER_CELL


PROJECT_FOLDER = Path(".")
COMBINED_DIR = PROJECT_FOLDER / "combined_normalized_trials"
OUTPUT_DIR = PROJECT_FOLDER / "group_plots"


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value).strip())


def parse_combined_filename(path):
    """
    Example:
    NGexp9-light-40Hz-eeg_rel_delta-n11.csv
    """
    pattern = r"^(?P<mouse>.+?)-(?P<anes>.+?)-(?P<freq>\d+)Hz-(?P<metric>.+)-n(?P<n_trials>\d+)\.csv$"
    match = re.match(pattern, path.name)

    if not match:
        return None

    info = match.groupdict()
    info["freq"] = int(info["freq"])
    info["n_trials"] = int(info["n_trials"])
    return info


def load_combined_files():
    rows = []

    if not COMBINED_DIR.exists():
        print("No combined_normalized_trials folder found.")
        return pd.DataFrame()

    for path in COMBINED_DIR.glob("*.csv"):
        info = parse_combined_filename(path)

        if info is None:
            print(f"Skipping file with unexpected name: {path.name}")
            continue

        if info["n_trials"] < MIN_TRIALS_PER_CELL:
            # A cell built from one or two surviving trials carries the same weight as a
            # ten-trial cell once it reaches the group mean. Keep it out. See issue #12.
            print(
                f"Excluding {path.name}: {info['n_trials']} good trial(s), "
                f"minimum is {MIN_TRIALS_PER_CELL}"
            )
            continue

        df = pd.read_csv(path)

        if "time_from_stim" not in df.columns or "mean" not in df.columns:
            print(f"Skipping {path.name}: missing time_from_stim or mean")
            continue

        temp = df[["time_from_stim", "mean"]].copy()
        temp["mouse"] = info["mouse"]
        temp["anes"] = info["anes"]
        temp["frequency_hz"] = info["freq"]
        temp["metric"] = info["metric"]
        temp["n_trials"] = info["n_trials"]

        rows.append(temp)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def metric_title(metric):
    names = {
        "eeg_rel_delta": "EEG rel_delta",
        "eeg_supp_mask_probability": "EEG supp_mask probability",
        "pupil_diameter_mm": "Pupil diameter",
        "heart_rate_hr_bpm": "Heart rate",
        "respiration_resp_bpm": "Respiration rate",
    }
    return names.get(metric, metric)


def y_label(metric):
    if metric == "eeg_supp_mask_probability":
        return "Suppression probability"
    return "Change from baseline (%)"


def draw_group_plots(data):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Missing matplotlib. Run: pip install matplotlib")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if data.empty:
        print("No data to plot.")
        return

    for (metric, anes), group in data.groupby(["metric", "anes"]):
        plt.figure(figsize=(10, 5))

        title = metric_title(metric)
        all_mice = sorted(group["mouse"].unique())
        total_n_mice = len(all_mice)

        for freq, freq_df in group.groupby("frequency_hz"):
            # Each mouse contributes one mean curve.
            summary = (
                freq_df.groupby("time_from_stim", as_index=False)
                .agg(
                    group_mean=("mean", "mean"),
                    group_sd=("mean", "std"),
                    n_mice=("mouse", "nunique"),
                    n_trials=("n_trials", "sum"),
                )
                .sort_values("time_from_stim")
            )

            n_mice_freq = int(summary["n_mice"].max())
            n_trials_freq = int(summary["n_trials"].max())

            label = f"{int(freq)} Hz (n={n_mice_freq} mice, {n_trials_freq} trials)"

            plt.plot(
                summary["time_from_stim"],
                summary["group_mean"],
                linewidth=2.5,
                label=label,
            )

            # With a single mouse the SD is undefined. Drawing it as zero makes the least
            # certain curve look like the most certain one, so draw no band at all.
            if n_mice_freq > 1:
                summary["group_sd"] = summary["group_sd"].fillna(0)
                plt.fill_between(
                    summary["time_from_stim"],
                    summary["group_mean"] - summary["group_sd"],
                    summary["group_mean"] + summary["group_sd"],
                    alpha=0.18,
                )

        plt.axhline(0, linewidth=0.8, alpha=0.6)
        plt.axvline(0, linewidth=0.8, alpha=0.6)
        plt.xlim(-10, 60)

        plt.xlabel("Time from stimulus onset (s)")
        plt.ylabel(y_label(metric))
        plt.title(f"{title} - {anes} - n={total_n_mice} mice")
        plt.legend()
        plt.tight_layout()

        out_name = f"group_{safe_name(metric)}_{safe_name(anes)}.png"
        plt.savefig(OUTPUT_DIR / out_name, dpi=200)
        plt.close()

        print(f"Saved: {out_name}")


if __name__ == "__main__":
    data = load_combined_files()
    draw_group_plots(data)
    print("Done group plots.")