from pathlib import Path

from segment_alpha_recording import run_one_recording


# Change only these three lines for each recording:
MOUSE = "NGexp9"
ANES = "light"
RECORDING = "NGexp9-260607-162633_pupil_anesthesia-260705-130101"


PROJECT_FOLDER = Path(".")

# Change these only if your folder names are different:
PUPIL_MAIN_FOLDER = PROJECT_FOLDER / "ally_pupil"
RESULTS_MAIN_FOLDER = PROJECT_FOLDER / "ally_results"

PUPIL_FOLDER = PUPIL_MAIN_FOLDER / RECORDING
RESULTS_FOLDER = RESULTS_MAIN_FOLDER / RECORDING

STIM = RESULTS_FOLDER / f"stim_trains_{RECORDING}.csv"
PUPIL = PUPIL_FOLDER / f"pupil_response_full_{RECORDING}.csv"
HR_RR = RESULTS_FOLDER / f"hr_rr_10hz_{RECORDING}.csv"
EEG = RESULTS_FOLDER / f"eeg_full_{RECORDING}_sw_1000Hz.csv"


if __name__ == "__main__":
    run_one_recording(
        folder=PROJECT_FOLDER,
        mouse=MOUSE,
        anes=ANES,
        recording=RECORDING,
        stim_path=STIM,
        pupil_path=PUPIL,
        hr_rr_path=HR_RR,
        eeg_path=EEG,
    )