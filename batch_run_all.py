from pathlib import Path

import pandas as pd

from quality_control import MIN_TRIALS_PER_CELL
from segment_alpha_recording import run_one_recording


PROJECT_FOLDER = Path(".")

ALPHA_RECORDINGS = PROJECT_FOLDER / "alpha_recordings - july.csv"

PUPIL_MAIN_FOLDER = PROJECT_FOLDER / "ally_pupil"
RESULTS_MAIN_FOLDER = PROJECT_FOLDER / "ally_results"

COMBINED_DIR = PROJECT_FOLDER / "combined_normalized_trials"


METRICS = [
    {
        "file": "eeg_segmented_trials.csv",
        "percent_col": "rel_delta_percent_change_from_baseline",
        "label": "eeg_rel_delta",
    },
    {
        "file": "eeg_segmented_trials.csv",
        "percent_col": "supp_mask_probability",
        "label": "eeg_supp_mask_probability",
    },
    {
        "file": "pupil_segmented_trials.csv",
        "percent_col": "diameter_mm_percent_change_from_baseline",
        "label": "pupil_diameter_mm",
    },
    {
        "file": "hr_rr_segmented_trials.csv",
        "percent_col": "hr_bpm_percent_change_from_baseline",
        "label": "heart_rate_hr_bpm",
    },
    {
        "file": "hr_rr_segmented_trials.csv",
        "percent_col": "resp_bpm_percent_change_from_baseline",
        "label": "respiration_resp_bpm",
    },
]


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value).strip())


def find_trial_col(df):
    for col in ["trial_id", "stim_id", "stim_index", "stim_number"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find trial column")


def find_recording_folder(mouse, filename):
    """
    alpha_recordings has filename like:
    pupil_anesthesia-260705-130101

    But the real folder is usually like:
    NGexp9-260607-162633_pupil_anesthesia-260705-130101
    """
    candidates = []
    for folder in RESULTS_MAIN_FOLDER.iterdir():
        if not folder.is_dir():
            continue
        if folder.name.startswith(str(mouse)) and folder.name.endswith(str(filename)):
            candidates.append(folder.name)

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Could not find folder for mouse={mouse}, filename={filename} inside {RESULTS_MAIN_FOLDER}"
        )

    if len(candidates) > 1:
        raise ValueError(
            f"Found more than one folder for mouse={mouse}, filename={filename}: {candidates}"
        )

    return candidates[0]

def find_first_file(folder, patterns):
    for pattern in patterns:
        matches = list(folder.glob(pattern))
        if matches:
            return matches[0]
    return None

def run_all_recordings():
    alpha = pd.read_csv(ALPHA_RECORDINGS)

    for _, row in alpha.iterrows():
        mouse = str(row["mouse"]).strip()
        anes = str(row["anes"]).strip()
        filename = str(row["filename"]).strip()

        recording = find_recording_folder(mouse, filename)

        print("\n==============================")
        print(f"Running: {mouse} {anes} {recording}")
        print("==============================")

        pupil_folder = PUPIL_MAIN_FOLDER / recording
        results_folder = RESULTS_MAIN_FOLDER / recording

        stim_path = find_first_file(results_folder, [
            f"stim_trains_{recording}.csv",
            "stim_trains_*.csv",
        ])

        pupil_path = find_first_file(pupil_folder, [
            f"pupil_response_full_{recording}.csv",
            "pupil_response_full_*.csv",
        ])

        hr_rr_path = find_first_file(results_folder, [
            f"hr_rr_10hz_{recording}.csv",
            "hr_rr_10hz_*.csv",
            "hr_rr_*.csv",
        ])

        eeg_path = find_first_file(results_folder, [
            f"eeg_full_{recording}_sw_1000Hz.csv",
            "eeg_full_*_sw_1000Hz.csv",
            "eeg_full_*.csv",
        ])

        if stim_path is None:
            print("Skipping because missing stim file in:", results_folder)
            continue

        if pupil_path is None:
            print("Missing pupil, will skip pupil:", pupil_folder)
        if hr_rr_path is None:
            print("Missing hr_rr, will skip hr_rr:", results_folder)
        if eeg_path is None:
            print("Missing eeg, will skip eeg:", results_folder)

        nominal_duration_s = pd.to_numeric(row.get("duration_s"), errors="coerce")

        run_one_recording(
            folder=PROJECT_FOLDER,
            mouse=mouse,
            anes=anes,
            recording=recording,
            stim_path=stim_path,
            pupil_path=pupil_path,
            hr_rr_path=hr_rr_path,
            eeg_path=eeg_path,
            nominal_duration_s=nominal_duration_s,
        )


def combine_same_mouse_same_anes():
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    output_folders = [
        folder
        for folder in PROJECT_FOLDER.glob("segmented_output_*")
        if folder.is_dir()
    ]

    if not output_folders:
        print("No segmented_output folders found.")
        return

    for metric in METRICS:
        all_rows = []

        for folder in output_folders:
            path = folder / metric["file"]
            if not path.exists():
                continue

            df = pd.read_csv(path)

            if metric["percent_col"] not in df.columns:
                print(f"Skipping {folder.name} {metric['label']}: missing {metric['percent_col']}")
                continue

            needed = ["mouse", "anes", "recording", "nominal_frequency_hz", "time_from_stim", "is_good_stim"]
            missing_cols = [c for c in needed if c not in df.columns]
            if missing_cols:
                print(f"Skipping {path}: missing columns {missing_cols}")
                continue

            trial_col = find_trial_col(df)

            df = df[df["is_good_stim"].astype(bool)].copy()
            if df.empty:
                continue

            df["recording_short"] = df["recording"].astype(str).str.split("pupil_anesthesia-").str[-1]
            df["trial_unique"] = df["recording_short"].astype(str) + "_" + df[trial_col].astype(str)

            keep = [
                "mouse",
                "anes",
                "nominal_frequency_hz",
                "time_from_stim",
                "trial_unique",
                metric["percent_col"],
            ]
            all_rows.append(df[keep])

        if not all_rows:
            print(f"No data for {metric['label']}")
            continue

        data = pd.concat(all_rows, ignore_index=True)
        data = data.dropna(subset=[metric["percent_col"], "time_from_stim", "nominal_frequency_hz"])

        for (mouse, anes, freq), group in data.groupby(["mouse", "anes", "nominal_frequency_hz"]):
            freq = int(freq)

            pivot = group.pivot_table(
                index="time_from_stim",
                columns="trial_unique",
                values=metric["percent_col"],
                aggfunc="mean",
            ).sort_index()

            n_trials = pivot.shape[1]

            if n_trials < MIN_TRIALS_PER_CELL:
                print(
                    f"Excluding {mouse} {anes} {freq}Hz {metric['label']}: "
                    f"{n_trials} good trial(s), minimum is {MIN_TRIALS_PER_CELL}"
                )
                continue

            pivot["mean"] = pivot.mean(axis=1)
            pivot["std"] = pivot.drop(columns=["mean"]).std(axis=1)

            out_name = (
                f"{safe_name(mouse)}-{safe_name(anes)}-"
                f"{freq}Hz-{safe_name(metric['label'])}-n{n_trials}.csv"
            )

            pivot.reset_index().to_csv(COMBINED_DIR / out_name, index=False)
            print(f"Saved combined: {out_name}")


if __name__ == "__main__":
    run_all_recordings()
    combine_same_mouse_same_anes()
    print("\nDone all recordings + combined normalized files.")