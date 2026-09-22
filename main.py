from pathlib import Path

from segment_alpha_recording import (
    extract_windows,
    infer_recording_row,
    prepare_stim_table,
    read_csv,
    save_outputs,
)

FOLDER = Path(".")

ALPHA_RECORDINGS = FOLDER / "alpha_recordings.csv"
STIM = FOLDER / "stim_trains_NGexp9-260607-162633_pupil_anesthesia-260630-152605.csv"
PUPIL = FOLDER / "pupil_response_full_NGexp9-260607-162633_pupil_anesthesia-260630-152605.csv"
HR_RR = FOLDER / "hr_rr_10hz_NGexp9-260607-162633_pupil_anesthesia-260630-152605.csv"
EEG = FOLDER / "eeg_full_NGexp9-260607-162633_pupil_anesthesia-260630-152605_sw_1000Hz.csv"

OUTPUT_DIR = FOLDER / "segmented_output"

PRE_SECONDS = 10
POST_SECONDS = 60


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    alpha_recordings = read_csv(ALPHA_RECORDINGS)
    stim = prepare_stim_table(read_csv(STIM))
    recording_row = infer_recording_row(alpha_recordings, None, [STIM, PUPIL, HR_RR, EEG])

    if recording_row is not None:
        recording_row.to_frame().T.to_csv(OUTPUT_DIR / "recording_metadata.csv", index=False)

    stim.to_csv(OUTPUT_DIR / "stim_metadata.csv", index=False)

    reports = []
    for modality, path in [("pupil", PUPIL), ("hr_rr", HR_RR), ("eeg", EEG)]:
        segmented, report = extract_windows(
            data=read_csv(path),
            stim=stim,
            recording_row=recording_row,
            modality=modality,
            pre_seconds=PRE_SECONDS,
            post_seconds=POST_SECONDS,
        )
        save_outputs(OUTPUT_DIR, modality, segmented, write_individual_trials=True)
        reports.append(report)
        print(f"{modality}: saved {len(segmented)} rows")

    import pandas as pd
    pd.concat(reports, ignore_index=True).to_csv(OUTPUT_DIR / "segmentation_report.csv", index=False)

    print("Done! Results are in segmented_output")


if __name__ == "__main__":
    main()