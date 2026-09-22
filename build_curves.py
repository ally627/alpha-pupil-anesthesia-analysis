"""Segment every recording from the raw files into one tidy table of per-trial curves.

Replaces the per-recording `segmented_output_*` trees for analysis purposes. Instead of
writing gigabytes of wide CSVs with the recording metadata repeated on every sample, this
produces a single long table:

    recording, mouse, anes, trial_id, nominal_frequency_hz, metric, time_from_stim, value

`value` is already baseline-corrected in the unit that suits the metric (percent change,
absolute change, or raw) — see `derived_metrics.METRICS`.

Two things differ from the original pipeline beyond the units:

* **The pre-stimulus window is 20 s, not 10 s.** HRV at time t is computed from the beats
  in [t-10, t], so a baseline point at -9.5 s previously had half a window of history and
  the curve ramped up from an artefact rather than starting at the animal's real level.
  The extra 10 s only fills the window; the analysed range stays [-10, +60]. Inter-stimulus
  interval is 120 s, so nothing overlaps.
* **Samples are binned onto a fixed 0.1 s grid** instead of rounding `time_from_stim` and
  pivoting on the rounded float. See issue #5.

Usage:
    python build_curves.py --raw raw/ally_results --recordings "data/alpha_recordings - july.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from derived_metrics import (
    METRICS,
    Unit,
    add_bsr,
    add_hrv,
    baseline_correct,
    resample_to_grid,
)
from quality_control import annotate_stim_quality
from segment_alpha_recording import prepare_stim_table


PRE_SECONDS = 20.0          # cut this much before onset ...
POST_SECONDS = 60.0
ANALYSIS_START_S = -10.0    # ... but only report from here, once HRV has a full window.
GRID_STEP_S = 0.1


def find_one(folder: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]
    return None


def locate(raw_root: Path, recording: str) -> dict[str, Path | None]:
    results = raw_root / recording
    pupil = raw_root / "ally_pupil" / recording

    return {
        "stim": find_one(results, [f"stim_trains_{recording}.csv", "stim_trains_*.csv"]),
        "hr_rr": find_one(results, [f"hr_rr_10hz_{recording}.csv", "hr_rr_10hz_*.csv"]),
        "eeg": find_one(results, ["eeg_full_*_1000Hz.csv"]),
        # The plain file, not *_normalized_*: normalisation there is against the whole
        # recording, and every metric here is normalised per trial instead.
        "pupil": find_one(pupil, [f"pupil_response_full_{recording}.csv"]),
    }


def cut_trials(data: pd.DataFrame, stim: pd.DataFrame) -> pd.DataFrame:
    """Window the recording around every good stimulus. Uses searchsorted, not a mask scan."""
    data = data.sort_values("time").reset_index(drop=True)
    times = data["time"].to_numpy()

    pieces = []
    for _, row in stim.iterrows():
        if not bool(row["is_good_stim"]):
            continue

        onset = float(row["stim_time"])
        lo, hi = np.searchsorted(times, [onset - PRE_SECONDS, onset + POST_SECONDS])
        piece = data.iloc[lo:hi].copy()
        if piece.empty:
            continue

        piece["trial_id"] = row["trial_id"]
        piece["nominal_frequency_hz"] = int(row["nominal_frequency_hz"])
        piece["time_from_stim"] = piece["time"] - onset
        pieces.append(piece)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def build_recording(raw_root: Path, recording: str, mouse: str, anes: str,
                    nominal_duration_s: float | None) -> pd.DataFrame:
    paths = locate(raw_root, recording)

    if paths["stim"] is None:
        print(f"{recording}: no stim file, skipped")
        return pd.DataFrame()

    stim = prepare_stim_table(pd.read_csv(paths["stim"]), nominal_duration_s=nominal_duration_s)

    frames = {}

    if paths["hr_rr"] is not None:
        hr = cut_trials(pd.read_csv(paths["hr_rr"]), stim)
        if not hr.empty:
            frames["hr_rr"] = add_hrv(hr)

    if paths["pupil"] is not None:
        pupil = cut_trials(
            pd.read_csv(paths["pupil"], usecols=lambda c: c in
                        {"time", "diameter_mm", "is_interpolated"}),
            stim,
        )
        if not pupil.empty:
            frames["pupil"] = pupil

    if paths["eeg"] is not None:
        eeg = cut_trials(
            pd.read_csv(paths["eeg"], usecols=lambda c: c in
                        {"time", "rel_delta", "supp_mask", "psupp", "is_interpolated"}),
            stim,
        )
        if not eeg.empty:
            frames["eeg"] = add_bsr(eeg)

    rows = []

    for key, source, column, unit, _label in METRICS:
        frame = frames.get(source)
        if frame is None:
            continue

        # eeg_bsr is derived from supp_mask by add_bsr and lands in the `bsr` column.
        column = "bsr" if key == "eeg_bsr" else column
        if column not in frame.columns:
            print(f"{recording}: {key} unavailable (no column {column})")
            continue

        work = frame[["trial_id", "nominal_frequency_hz", "time_from_stim", column]].copy()
        work["value"] = baseline_correct(work, column, unit)

        binned = resample_to_grid(work, "value", step_s=GRID_STEP_S)
        frequencies = work.groupby("trial_id")["nominal_frequency_hz"].first()
        binned["nominal_frequency_hz"] = binned["trial_id"].map(frequencies)

        binned = binned[binned["time_from_stim"] >= ANALYSIS_START_S]
        binned["metric"] = key
        rows.append(binned)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["recording"] = recording
    out["mouse"] = mouse
    out["anes"] = anes

    n_trials = out.groupby("nominal_frequency_hz")["trial_id"].nunique().to_dict()
    print(f"{recording}: {len(out):,} rows, good trials by frequency {n_trials}")

    return out[["recording", "mouse", "anes", "trial_id", "nominal_frequency_hz",
                "metric", "time_from_stim", "value"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("raw/ally_results"))
    parser.add_argument("--recordings", type=Path, default=Path("data/alpha_recordings - july.csv"))
    parser.add_argument("--output", type=Path, default=Path("curves_per_trial.csv"))
    args = parser.parse_args()

    alpha = pd.read_csv(args.recordings)

    folders = {p.name for p in args.raw.iterdir() if p.is_dir() and p.name != "ally_pupil"}

    all_rows = []

    for _, row in alpha.iterrows():
        mouse = str(row["mouse"]).strip()
        filename = str(row["filename"]).strip()

        matches = [f for f in folders if f.startswith(f"{mouse}-") and f.endswith(filename)]
        if len(matches) != 1:
            print(f"{mouse} {filename}: expected one folder, found {matches}")
            continue

        built = build_recording(
            raw_root=args.raw,
            recording=matches[0],
            mouse=mouse,
            anes=str(row["anes"]).strip(),
            nominal_duration_s=pd.to_numeric(row.get("duration_s"), errors="coerce"),
        )
        if not built.empty:
            all_rows.append(built)

    if not all_rows:
        print("Nothing built.")
        return

    curves = pd.concat(all_rows, ignore_index=True)
    curves.to_csv(args.output, index=False)
    print(f"\nWrote {len(curves):,} rows to {args.output}")
    print(curves.groupby(["metric"])["value"].describe()[["count", "mean", "std"]])


if __name__ == "__main__":
    main()
