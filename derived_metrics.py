"""Metrics derived from the segmented tables, and the unit each one is reported in.

Three ideas live here.

1. **Not every metric is a percent change.** `rel_delta` is already delta power as a
   fraction of total power, and `supp_mask` is a probability. Expressing either as a
   percent change from baseline is a ratio of a ratio — hard to read, and unstable when
   the baseline is small. Those are reported as an absolute change instead, in the units
   they already have. See issues #3 and #15.

2. **HRV.** Heart rate says how fast the heart beats; heart rate variability says how
   regular it is, and it responds to arousal on its own terms. Computed as the standard
   deviation of the inter-beat interval over a sliding window (SDNN), from the
   interpolation-corrected `hr_ibi`, never from `hr_ibi_raw`. See issue #16.

3. **Burst suppression.** `supp_mask` is 0/1 per sample; the fraction of time it is 1 over
   a window is the burst-suppression ratio, the standard read-out of anesthetic depth.
   See issue #17.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Sliding window for SDNN. The textbook figure is 60 s, but the stimulus response in this
# experiment lives in the first ten seconds, so a 60 s window would smear it out entirely.
# Report this as SDNN-10s, not as SDNN.
HRV_WINDOW_S = 10.0

# Fraction of interpolated beats above which an HRV window is discarded. Interpolated beats
# are smooth by construction and drag variability towards zero.
HRV_MAX_INTERPOLATED = 0.25

# Window for the burst-suppression ratio, in seconds.
BSR_WINDOW_S = 5.0

BASELINE_START_S = -10.0
BASELINE_END_S = 0.0


class Unit:
    PERCENT_CHANGE = "percent_change"      # (x - baseline) / baseline * 100
    ABSOLUTE_CHANGE = "absolute_change"    # x - baseline, in the metric's own units
    RAW = "raw"                            # reported as is, no baseline correction


METRICS = [
    # key                  source     column          unit                    axis label
    ("pupil_diameter_mm",  "pupil",   "diameter_mm",  Unit.PERCENT_CHANGE,    "Change from baseline (%)"),
    ("heart_rate_hr_bpm",  "hr_rr",   "hr_bpm",       Unit.PERCENT_CHANGE,    "Change from baseline (%)"),
    ("respiration_resp_bpm", "hr_rr", "resp_bpm",     Unit.PERCENT_CHANGE,    "Change from baseline (%)"),
    ("hrv_sdnn_ms",        "hr_rr",   "hrv_sdnn_ms",  Unit.ABSOLUTE_CHANGE,   "Change from baseline (ms)"),
    ("rr_variability_ms",  "hr_rr",   "rr_sdnn_ms",   Unit.ABSOLUTE_CHANGE,   "Change from baseline (ms)"),
    ("eeg_rel_delta",      "eeg",     "rel_delta",    Unit.ABSOLUTE_CHANGE,   "Change from baseline (fraction)"),
    ("eeg_bsr",            "eeg",     "supp_mask",    Unit.RAW,               "Burst-suppression ratio (0-1)"),
]

METRIC_BY_KEY = {m[0]: m for m in METRICS}


def add_hrv(hr_rr: pd.DataFrame, window_s: float = HRV_WINDOW_S) -> pd.DataFrame:
    """Add `hrv_sdnn_ms` and `rr_sdnn_ms`: rolling SD of the inter-beat interval, in ms.

    Operates per trial, on the time axis, using the corrected IBI columns. The IBI series
    is sampled at the same rate as everything else (it repeats between beats), so the
    rolling SD is taken over unique successive interval values rather than over samples —
    otherwise a slow heart rate would inflate the window's apparent beat count.
    """
    hr_rr = hr_rr.copy()

    for source, target, interpolated_flag in [
        ("hr_ibi", "hrv_sdnn_ms", "hr_is_interpolated"),
        ("rr_ibi", "rr_sdnn_ms", "rr_is_interpolated"),
    ]:
        if source not in hr_rr.columns:
            hr_rr[target] = np.nan
            continue

        values = []
        for _, trial in hr_rr.groupby("trial_id", sort=False):
            trial = trial.sort_values("time_from_stim")
            ibi = pd.to_numeric(trial[source], errors="coerce")
            time = pd.to_numeric(trial["time_from_stim"], errors="coerce")

            # One sample per beat: keep the first sample of each new interval value.
            new_beat = ibi.ne(ibi.shift())
            beats = pd.DataFrame({"time": time[new_beat], "ibi": ibi[new_beat]})

            if interpolated_flag in trial.columns:
                beats["interpolated"] = (
                    trial.loc[new_beat, interpolated_flag].astype(str).str.lower().eq("true").to_numpy()
                )
            else:
                beats["interpolated"] = False

            sdnn = []
            for t in time:
                window = beats[(beats["time"] > t - window_s) & (beats["time"] <= t)]
                if len(window) < 3 or window["interpolated"].mean() > HRV_MAX_INTERPOLATED:
                    sdnn.append(np.nan)
                else:
                    sdnn.append(window["ibi"].std(ddof=1) * 1000.0)  # seconds -> ms

            values.append(pd.Series(sdnn, index=time.index))

        hr_rr[target] = pd.concat(values).reindex(hr_rr.index) if values else np.nan

    return hr_rr


def add_bsr(eeg: pd.DataFrame, window_s: float = BSR_WINDOW_S) -> pd.DataFrame:
    """Add `bsr`: fraction of the preceding `window_s` spent in suppression, per trial."""
    eeg = eeg.copy()
    mask = pd.to_numeric(eeg["supp_mask"], errors="coerce")

    out = []
    for _, trial in eeg.groupby("trial_id", sort=False):
        trial = trial.sort_values("time_from_stim")
        time = pd.to_numeric(trial["time_from_stim"], errors="coerce")
        dt = float(np.median(np.diff(time.to_numpy()))) if len(trial) > 1 else 1.0
        n = max(int(round(window_s / dt)), 1)
        out.append(mask.loc[trial.index].rolling(n, min_periods=max(n // 2, 1)).mean())

    eeg["bsr"] = pd.concat(out).reindex(eeg.index)
    return eeg


def baseline_correct(trial_curves: pd.DataFrame, value_col: str, unit: str,
                     trial_col: str = "trial_id") -> pd.Series:
    """Return the baseline-corrected series for one metric, according to its unit.

    The baseline is the median of the value over [-10, 0) seconds, per trial — robust to a
    single bad sample in the pre-stimulus window, which a mean is not.
    """
    values = pd.to_numeric(trial_curves[value_col], errors="coerce")
    time = pd.to_numeric(trial_curves["time_from_stim"], errors="coerce")

    if unit == Unit.RAW:
        return values

    in_baseline = (time >= BASELINE_START_S) & (time < BASELINE_END_S)
    baseline = values[in_baseline].groupby(trial_curves.loc[in_baseline, trial_col]).median()
    per_sample_baseline = trial_curves[trial_col].map(baseline)

    if unit == Unit.ABSOLUTE_CHANGE:
        return values - per_sample_baseline

    # Percent change. Guard against a baseline so close to zero that the ratio explodes:
    # require it to be at least 1% of the metric's own median magnitude in this recording.
    scale = values.abs().median()
    usable = per_sample_baseline.abs() > max(scale * 0.01, np.finfo(float).eps)
    return ((values - per_sample_baseline) / per_sample_baseline.where(usable)) * 100.0


def resample_to_grid(df: pd.DataFrame, value_col: str, step_s: float = 0.1,
                     trial_col: str = "trial_id") -> pd.DataFrame:
    """Bin samples onto a fixed grid of `step_s`, per trial.

    Replaces the previous approach of rounding `time_from_stim` and pivoting on the rounded
    float, which silently split or merged samples depending on where the stimulus fell
    between two samples. See issue #5.
    """
    out = df[[trial_col, "time_from_stim", value_col]].copy()
    out["time_from_stim"] = (
        (pd.to_numeric(out["time_from_stim"], errors="coerce") / step_s).round() * step_s
    ).round(3)
    return (
        out.groupby([trial_col, "time_from_stim"], as_index=False)[value_col]
        .mean()
    )
