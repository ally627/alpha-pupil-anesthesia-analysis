#!/usr/bin/env python3
"""
Segment one Alpha anesthesia recording into stimulus-locked windows.

This script is tailored to the current experiment files:
- alpha_recordings.csv: recording metadata by mouse and anesthesia condition.
- stim_trains_*.csv: exact stimulus start/end times and stimulation details.
- pupil_response_full_*.csv: pupil time series.
- hr_rr_*.csv: heart rate and respiration time series.
- eeg_full_*.csv: EEG time series.

For every stimulus, it extracts -10 seconds to +60 seconds around stimulus onset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from quality_control import (
    MIN_TRIALS_PER_CELL,
    annotate_stim_quality,
    summarise_rejections,
)


DEFAULT_PRE_SECONDS = 10.0
DEFAULT_POST_SECONDS = 60.0


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def infer_recording_row(
    alpha_recordings: pd.DataFrame,
    recording_filename: str | None,
    data_paths: list[Path],
) -> pd.Series | None:
    if alpha_recordings.empty or "filename" not in alpha_recordings.columns:
        return None

    if recording_filename:
        matches = alpha_recordings[
            alpha_recordings["filename"].astype(str).eq(recording_filename)
            | alpha_recordings["filename"].astype(str).str.contains(recording_filename, regex=False)
        ]
    else:
        joined_names = " ".join(path.name for path in data_paths)
        matches = alpha_recordings[
            alpha_recordings["filename"].astype(str).apply(lambda name: name in joined_names)
        ]

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        filenames = ", ".join(matches["filename"].astype(str).tolist())
        raise ValueError(f"More than one alpha_recordings row matched: {filenames}")
    return matches.iloc[0]


def prepare_stim_table(stim: pd.DataFrame, nominal_duration_s: float | None = None) -> pd.DataFrame:
    required = {"stim_index", "start_time_s", "end_time_s"}
    missing = required - set(stim.columns)
    if missing:
        raise ValueError(f"Missing required columns in stim file: {sorted(missing)}")

    stim = stim.copy()
    stim["trial_id"] = stim["stim_index"].apply(lambda value: f"stim_{int(value):02d}")
    stim["stim_time"] = stim["start_time_s"].astype(float)
    stim["stim_end_time"] = stim["end_time_s"].astype(float)
    stim["stim_duration_s"] = stim["stim_end_time"] - stim["stim_time"]

    if "frequency" in stim.columns:
        stim = stim.rename(columns={"frequency": "actual_frequency_hz"})

    # The condition label comes from the rate actually delivered (pulses / duration),
    # not from the pulse count alone: the stimulator sometimes spread the right number
    # of pulses over the wrong interval, which makes a "100 Hz" train a real 42 Hz one.
    stim = annotate_stim_quality(stim, nominal_duration_s=nominal_duration_s)

    rejected = int((~stim["is_good_stim"]).sum())
    if rejected:
        by_reason = stim.loc[~stim["is_good_stim"], "reject_reason"].value_counts().to_dict()
        print(f"Stimulus QC: rejected {rejected} of {len(stim)} trains {by_reason}")

    return stim.sort_values("stim_time").reset_index(drop=True)


def add_metadata_columns(
    trial: pd.DataFrame,
    stim_row: pd.Series,
    recording_row: pd.Series | None,
    modality: str,
) -> pd.DataFrame:
    trial = trial.copy()
    trial.insert(0, "modality", modality)
    trial.insert(1, "trial_id", stim_row["trial_id"])
    trial.insert(2, "stim_index", int(stim_row["stim_index"]))
    trial.insert(3, "stim_time", float(stim_row["stim_time"]))
    trial.insert(4, "time_from_stim", trial["time"] - float(stim_row["stim_time"]))

    stim_metadata_cols = [
        "stim_end_time",
        "stim_duration_s",
        "train_index",
        "pulse_count",
        "actual_frequency_hz",
        "measured_frequency_hz",
        "nominal_frequency_hz",
        "duration_ok",
        "frequency_ok",
        "is_good_stim",
        "reject_reason",
    ]
    for col in stim_metadata_cols:
        if col in stim_row.index and col not in trial.columns:
            trial[col] = stim_row[col]

    if recording_row is not None:
        for col in ["mouse", "date", "anes", "filename", "amp", "isi_s", "min_iso", "max_iso"]:
            if col in recording_row.index and col not in trial.columns:
                trial[col] = recording_row[col]

    return trial


def extract_windows(
    data: pd.DataFrame,
    stim: pd.DataFrame,
    recording_row: pd.Series | None,
    modality: str,
    pre_seconds: float,
    post_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "time" not in data.columns:
        raise ValueError(f"Missing time column in {modality} file")

    data = data.copy()
    data["time"] = data["time"].astype(float)
    data = data.sort_values("time").reset_index(drop=True)

    recording_start = float(data["time"].min())
    recording_end = float(data["time"].max())
    trials = []
    report_rows = []

    for _, stim_row in stim.iterrows():
        stim_time = float(stim_row["stim_time"])
        start_time = stim_time - pre_seconds
        end_time = stim_time + post_seconds
        mask = (data["time"] >= start_time) & (data["time"] <= end_time)
        trial = data.loc[mask].copy()

        report_rows.append(
            {
                "modality": modality,
                "trial_id": stim_row["trial_id"],
                "stim_index": int(stim_row["stim_index"]),
                "stim_time": stim_time,
                "requested_start": start_time,
                "requested_end": end_time,
                "samples_found": len(trial),
                "has_full_pre_window": start_time >= recording_start,
                "has_full_post_window": end_time <= recording_end,
            }
        )

        if not trial.empty:
            trials.append(add_metadata_columns(trial, stim_row, recording_row, modality))

    if trials:
        segmented = pd.concat(trials, ignore_index=True)
    else:
        segmented = pd.DataFrame()

    return segmented, pd.DataFrame(report_rows)


def save_outputs(
    output_dir: Path,
    modality: str,
    segmented: pd.DataFrame,
    write_individual_trials: bool,
) -> None:
    segmented_path = output_dir / f"{modality}_segmented_trials.csv"
    segmented.to_csv(segmented_path, index=False)

    if write_individual_trials and not segmented.empty:
        trials_dir = output_dir / "individual_trials" / modality
        trials_dir.mkdir(parents=True, exist_ok=True)
        for trial_id, trial in segmented.groupby("trial_id", sort=False):
            trial.to_csv(trials_dir / f"{trial_id}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment Alpha experiment files around stimulus onset.")
    parser.add_argument("--alpha-recordings", required=True, type=Path)
    parser.add_argument("--stim", required=True, type=Path)
    parser.add_argument("--pupil", required=True, type=Path)
    parser.add_argument("--hr-rr", required=True, type=Path)
    parser.add_argument("--eeg", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--recording-filename",
        default=None,
        help="Optional filename value from alpha_recordings.csv, for example pupil_anesthesia-260630-152605.",
    )
    parser.add_argument("--pre-seconds", type=float, default=DEFAULT_PRE_SECONDS)
    parser.add_argument("--post-seconds", type=float, default=DEFAULT_POST_SECONDS)
    parser.add_argument("--no-individual-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    alpha_recordings = read_csv(args.alpha_recordings)
    stim = prepare_stim_table(read_csv(args.stim))
    data_paths = [args.stim, args.pupil, args.hr_rr, args.eeg]
    recording_row = infer_recording_row(alpha_recordings, args.recording_filename, data_paths)

    if recording_row is None:
        print("Warning: no matching row found in alpha_recordings.csv; continuing without recording metadata.")
    else:
        metadata_path = args.output_dir / "recording_metadata.csv"
        recording_row.to_frame().T.to_csv(metadata_path, index=False)
        print(f"Matched recording: {recording_row.get('mouse', '')} {recording_row.get('anes', '')} {recording_row.get('filename', '')}")

    stim_metadata_path = args.output_dir / "stim_metadata.csv"
    stim.to_csv(stim_metadata_path, index=False)

    reports = []
    modality_inputs = [
        ("pupil", args.pupil),
        ("hr_rr", args.hr_rr),
        ("eeg", args.eeg),
    ]

    for modality, path in modality_inputs:
        data = read_csv(path)
        segmented, report = extract_windows(
            data=data,
            stim=stim,
            recording_row=recording_row,
            modality=modality,
            pre_seconds=args.pre_seconds,
            post_seconds=args.post_seconds,
        )
        save_outputs(
            output_dir=args.output_dir,
            modality=modality,
            segmented=segmented,
            write_individual_trials=not args.no_individual_files,
        )
        reports.append(report)
        print(f"{modality}: saved {len(segmented)} rows across {report['samples_found'].gt(0).sum()} trials")

    all_reports = pd.concat(reports, ignore_index=True)
    all_reports.to_csv(args.output_dir / "segmentation_report.csv", index=False)
    print(f"Saved outputs in: {args.output_dir}")

def generate_plots(output_dir, segmented_by_modality, recording_row=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    normalized_dir = output_dir / "normalized_trials"
    normalized_dir.mkdir(parents=True, exist_ok=True)

    def get_meta_value(key, default="unknown"):
        if recording_row is None:
            return default
        try:
            if hasattr(recording_row, "get"):
                value = recording_row.get(key, default)
            else:
                value = default
            if pd.isna(value):
                return default
            return str(value)
        except Exception:
            return default

    mouse = get_meta_value("mouse", "unknown_mouse")
    anes = get_meta_value("anes", "unknown_anes")

    # Cells dropped for having too few good trials; written out at the end.
    excluded = []

    def safe_name(x):
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(x))

    def find_trial_col(df):
        for col in ["trial_id", "stim_id", "stim_index", "stim_number"]:
            if col in df.columns:
                return col
        raise ValueError("Could not find trial column. Expected trial_id/stim_id/stim_index.")

    def find_duration_col(df):
        for col in ["stim_duration_s", "duration_s", "train_duration_s", "stim_duration"]:
            if col in df.columns:
                return col
        return None

    def add_good_stim_column(df):
        # prepare_stim_table already decided this, per stimulus, using both the train
        # duration and the rate actually delivered. Trust it when it is present.
        if "is_good_stim" in df.columns:
            df["is_good_stim"] = df["is_good_stim"].astype(bool)
            return df

        duration_col = find_duration_col(df)
        if duration_col is None:
            df["is_good_stim"] = True
        else:
            nominal = float(pd.to_numeric(df[duration_col], errors="coerce").median())
            df["is_good_stim"] = df[duration_col].between(nominal * 0.8, nominal * 1.2, inclusive="both")
        return df

    def add_percent_change(df, value_col, percent_col):
        if value_col == "supp_mask":
            df[percent_col] = pd.to_numeric(df[value_col], errors="coerce").round(2)
            return df

        trial_col = find_trial_col(df)
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df["time_from_stim"] = pd.to_numeric(df["time_from_stim"], errors="coerce")

        baseline_mask = (df["time_from_stim"] >= -10) & (df["time_from_stim"] < 0)

        baseline = (
            df.loc[baseline_mask]
            .groupby(trial_col)[value_col]
            .median()
            .replace(0, np.nan)
        )

        df["_baseline_value"] = df[trial_col].map(baseline)
        df[percent_col] = ((df[value_col] - df["_baseline_value"]) / df["_baseline_value"]) * 100
        df = df.drop(columns=["_baseline_value"])
        return df

    def make_summary_table(df, value_col, percent_col, modality_label):
        trial_col = find_trial_col(df)

        good = df[df["is_good_stim"].astype(bool)].copy()
        good = good.dropna(subset=[percent_col, "time_from_stim", "nominal_frequency_hz"])

        if good.empty:
            print(f"No good stim data for {modality_label}")
            return

        for frequency, freq_df in good.groupby("nominal_frequency_hz"):
            frequency = int(frequency)
            pivot = freq_df.pivot_table(
                index="time_from_stim",
                columns=trial_col,
                values=percent_col,
                aggfunc="mean",
            ).sort_index()

            pivot["mean"] = pivot.mean(axis=1)
            pivot["std"] = pivot.drop(columns=["mean"]).std(axis=1)

            n_trials = pivot.drop(columns=["mean", "std"]).shape[1]

            if n_trials < MIN_TRIALS_PER_CELL:
                # One or two surviving trials is not an estimate. Write it aside so the
                # exclusion is visible, and keep it out of the aggregation. See issue #12.
                excluded.append(
                    {
                        "mouse": mouse,
                        "anes": anes,
                        "frequency_hz": frequency,
                        "metric": modality_label,
                        "n_trials": n_trials,
                        "minimum_required": MIN_TRIALS_PER_CELL,
                    }
                )
                print(
                    f"Excluded cell {mouse} {anes} {frequency}Hz {modality_label}: "
                    f"only {n_trials} good trial(s), minimum is {MIN_TRIALS_PER_CELL}"
                )
                continue

            out_name = (
                f"{safe_name(mouse)}-{safe_name(anes)}-"
                f"{frequency}Hz-{safe_name(modality_label)}-n{n_trials}.csv"
            )
            pivot.reset_index().to_csv(normalized_dir / out_name, index=False)

    def plot_mean_only(df, percent_col, title, file_prefix, y_axis_label="Change from baseline (%)"):
        good = df[df["is_good_stim"].astype(bool)].copy()
        good = good.dropna(subset=[percent_col, "time_from_stim", "nominal_frequency_hz"])

        if good.empty:
            print(f"No good stim data for plot {title}")
            return

        plt.figure(figsize=(10, 5))

        for frequency, freq_df in good.groupby("nominal_frequency_hz"):
            summary = (
                freq_df.groupby("time_from_stim", as_index=False)[percent_col]
                .mean()
                .sort_values("time_from_stim")
            )

            plt.plot(
                summary["time_from_stim"],
                summary[percent_col],
                linewidth=2.5,
                label=f"{int(frequency)} Hz",
            )

        plt.axhline(0, linewidth=0.8, alpha=0.6)
        plt.axvline(0, linewidth=0.8, alpha=0.6)
        plt.xlim(-10, 60)
        plt.xlabel("Time from stimulus onset (s)")
        plt.ylabel(y_axis_label)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{safe_name(file_prefix)}_mean.png", dpi=200)
        plt.close()

    def plot_individual_plus_mean(df, percent_col, title, file_prefix, y_axis_label="Change from baseline (%)"):
        trial_col = find_trial_col(df)

        good = df[df["is_good_stim"].astype(bool)].copy()
        good = good.dropna(subset=[percent_col, "time_from_stim", "nominal_frequency_hz"])

        if good.empty:
            print(f"No good stim data for plot {title}")
            return

        plt.figure(figsize=(10, 5))

        for frequency, freq_df in good.groupby("nominal_frequency_hz"):
            for _, trial_df in freq_df.groupby(trial_col):
                trial_df = trial_df.sort_values("time_from_stim")
                plt.plot(
                    trial_df["time_from_stim"],
                    trial_df[percent_col],
                    linewidth=0.8,
                    alpha=0.25,
                )

            summary = (
                freq_df.groupby("time_from_stim", as_index=False)[percent_col]
                .mean()
                .sort_values("time_from_stim")
            )

            plt.plot(
                summary["time_from_stim"],
                summary[percent_col],
                linewidth=3,
                label=f"{int(frequency)} Hz mean",
            )

        plt.axhline(0, linewidth=0.8, alpha=0.6)
        plt.axvline(0, linewidth=0.8, alpha=0.6)
        plt.xlim(-10, 60)
        plt.xlabel("Time from stimulus onset (s)")
        plt.ylabel(y_axis_label)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{safe_name(file_prefix)}_individual_plus_mean.png", dpi=200)
        plt.close()

    configs = [
        {
            "modality": "eeg",
            "output_file": "eeg_segmented_trials.csv",
            "value_col": "rel_delta",
            "percent_col": "rel_delta_percent_change_from_baseline",
            "label": "eeg_rel_delta",
            "title": "EEG rel_delta",
            "y_axis_label": "Change from baseline (%)",
        },
        {
            "modality": "eeg",
            "output_file": "eeg_segmented_trials.csv",
            "value_col": "supp_mask",
            "percent_col": "supp_mask_probability",
            "label": "eeg_supp_mask",
            "title": "EEG supp_mask probability",
            # Not a percent change: supp_mask is a raw 0-1 probability. See issue #3.
            "y_axis_label": "Suppression probability (0-1)",
        },
        {
            "modality": "pupil",
            "output_file": "pupil_segmented_trials.csv",
            "value_col": "diameter_mm",
            "percent_col": "diameter_mm_percent_change_from_baseline",
            "label": "pupil_diameter_mm",
            "title": "Pupil diameter",
            "y_axis_label": "Change from baseline (%)",
        },
        {
            "modality": "hr_rr",
            "output_file": "hr_rr_segmented_trials.csv",
            "value_col": "hr_bpm",
            "percent_col": "hr_bpm_percent_change_from_baseline",
            "label": "heart_rate_hr_bpm",
            "title": "Heart rate",
            "y_axis_label": "Change from baseline (%)",
        },
        {
            "modality": "hr_rr",
            "output_file": "hr_rr_segmented_trials.csv",
            "value_col": "resp_bpm",
            "percent_col": "resp_bpm_percent_change_from_baseline",
            "label": "respiration_resp_bpm",
            "title": "Respiration rate",
            "y_axis_label": "Change from baseline (%)",
        },
    ]

    updated = {}

    for cfg in configs:
        modality = cfg["modality"]

        if modality not in segmented_by_modality:
            continue

        df = updated.get(modality, segmented_by_modality[modality]).copy()

        if cfg["value_col"] not in df.columns:
            print(f"Skipping {cfg['title']}: missing column {cfg['value_col']}")
            continue

        if modality == "eeg":
            df["time_from_stim"] = pd.to_numeric(df["time_from_stim"], errors="coerce").round(3)
        else:
            df["time_from_stim"] = pd.to_numeric(df["time_from_stim"], errors="coerce").round(1)

        df = add_good_stim_column(df)
        df = add_percent_change(df, cfg["value_col"], cfg["percent_col"])

        updated[modality] = df

        make_summary_table(df, cfg["value_col"], cfg["percent_col"], cfg["label"])

        plot_mean_only(
            df=df,
            percent_col=cfg["percent_col"],
            title=f"{cfg['title']} - {mouse} {anes}",
            file_prefix=cfg["label"],
            y_axis_label=cfg.get("y_axis_label", "Change from baseline (%)"),
        )

        plot_individual_plus_mean(
            df=df,
            percent_col=cfg["percent_col"],
            title=f"{cfg['title']} - {mouse} {anes} - individual trials + mean",
            file_prefix=f"{cfg['label']}_individual",
        )

    for modality, df in updated.items():
        if modality == "pupil":
            df.to_csv(output_dir / "pupil_segmented_trials.csv", index=False)
        elif modality == "hr_rr":
            df.to_csv(output_dir / "hr_rr_segmented_trials.csv", index=False)
        elif modality == "eeg":
            df.to_csv(output_dir / "eeg_segmented_trials.csv", index=False)

    if excluded:
        pd.DataFrame(excluded).to_csv(output_dir / "excluded_cells.csv", index=False)
        print(f"Excluded {len(excluded)} cell(s) with fewer than {MIN_TRIALS_PER_CELL} good trials")

    print(f"Saved plots in: {plots_dir}")
    print(f"Saved normalized trial tables in: {normalized_dir}")


def run_one_recording(
    folder,
    mouse,
    anes,
    recording,
    stim_path,
    pupil_path=None,
    hr_rr_path=None,
    eeg_path=None,
    nominal_duration_s=None,
):
    import pandas as pd

    recording_row = pd.Series({
        "mouse": mouse,
        "anes": anes,
        "recording": recording,
        "filename": recording,
    })

    output_dir = folder / f"segmented_output_{mouse}_{anes}_{recording.split('pupil_anesthesia-')[-1]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    stim = prepare_stim_table(read_csv(stim_path), nominal_duration_s=nominal_duration_s)
    stim.to_csv(output_dir / "stim_metadata.csv", index=False)
    summarise_rejections(stim).to_csv(output_dir / "stim_quality_summary.csv", index=False)
    recording_row.to_frame().T.to_csv(output_dir / "recording_metadata.csv", index=False)

    reports = []
    segmented_by_modality = {}

    modality_paths = [
        ("pupil", pupil_path),
        ("hr_rr", hr_rr_path),
        ("eeg", eeg_path),
    ]

    for modality, path in modality_paths:
        if path is None:
            print(f"{modality}: skipped because file is missing")
            continue

        if not path.exists():
            print(f"{modality}: skipped because file does not exist: {path}")
            continue

        segmented, report = extract_windows(
            data=read_csv(path),
            stim=stim,
            recording_row=recording_row,
            modality=modality,
            pre_seconds=10,
            post_seconds=60,
        )

        # make sure these columns exist for later combining
        segmented["mouse"] = mouse
        segmented["anes"] = anes
        segmented["recording"] = recording

        save_outputs(
            output_dir=output_dir,
            modality=modality,
            segmented=segmented,
            write_individual_trials=True,
        )

        segmented_by_modality[modality] = segmented
        reports.append(report)

        print(f"{modality}: saved {len(segmented)} rows")

    if reports:
        pd.concat(reports, ignore_index=True).to_csv(
            output_dir / "segmentation_report.csv",
            index=False
        )

    if segmented_by_modality:
        generate_plots(output_dir, segmented_by_modality, recording_row)

    print(f"Done. Results are in: {output_dir}")

if __name__ == "__main__":
    main()
