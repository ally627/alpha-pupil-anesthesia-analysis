"""Quality control for stimulus trains and for condition cells.

Two independent checks live here.

1. Per stimulus. A train is usable only if it lasted roughly the nominal duration
   AND actually delivered roughly the nominal frequency. The stimulator in this
   dataset sometimes emitted the correct number of pulses over the wrong interval,
   which turns a "100 Hz" train into a real 42 Hz train. Counting pulses alone
   cannot see that; dividing pulses by duration can.

2. Per condition cell. A (mouse, anesthesia, frequency) cell built from one or two
   trials is not an estimate, and averaging it into a group mean with the same
   weight as a ten-trial cell is how a single trial ends up steering a result.

See issues #2, #11 and #12.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# The two conditions the experiment delivers.
NOMINAL_FREQUENCIES_HZ = (40, 100)

# A train counts as its nominal frequency when the measured rate is within this
# fraction of it. 40 and 100 Hz are far apart, so 15% is generous and still leaves
# a wide rejection band (46-85 Hz) for malformed trains.
FREQUENCY_TOLERANCE = 0.15

# Accepted deviation from the nominal train duration recorded in alpha_recordings.csv.
DURATION_TOLERANCE = 0.20

# Minimum trials before a (mouse, anes, frequency) cell may contribute to a group mean.
MIN_TRIALS_PER_CELL = 3


def measured_frequency(pulse_count, duration_s):
    """Pulses actually delivered per second. NaN when the duration is unusable."""
    pulse_count = pd.to_numeric(pulse_count, errors="coerce")
    duration_s = pd.to_numeric(duration_s, errors="coerce")
    return pulse_count / duration_s.replace(0, np.nan)


def classify_frequency(measured_hz):
    """Snap a measured rate to 40 or 100 Hz, or to NaN when it is neither.

    Returns (nominal_hz, relative_error). A NaN nominal means the train did not
    deliver either condition and must not be analysed as if it had.
    """
    measured_hz = pd.to_numeric(measured_hz, errors="coerce")

    candidates = np.array(NOMINAL_FREQUENCIES_HZ, dtype=float)
    distances = np.abs(measured_hz.to_numpy()[:, None] - candidates[None, :])
    nearest = candidates[np.argmin(distances, axis=1)]

    relative_error = np.abs(measured_hz.to_numpy() - nearest) / nearest
    nominal = np.where(relative_error <= FREQUENCY_TOLERANCE, nearest, np.nan)

    return (
        pd.Series(nominal, index=measured_hz.index),
        pd.Series(relative_error, index=measured_hz.index),
    )


def annotate_stim_quality(stim: pd.DataFrame, nominal_duration_s: float | None = None) -> pd.DataFrame:
    """Add measured frequency, nominal frequency and per-check verdicts to a stim table.

    Columns added: measured_frequency_hz, frequency_error, nominal_frequency_hz,
    duration_ok, frequency_ok, is_good_stim, reject_reason.

    nominal_duration_s comes from alpha_recordings.csv. When it is not supplied the
    median observed duration is used, which is right when most trains are well formed
    and is reported in the rejection summary either way.
    """
    stim = stim.copy()

    if "stim_duration_s" not in stim.columns:
        raise ValueError("annotate_stim_quality needs stim_duration_s; call prepare_stim_table first")

    if nominal_duration_s is None or not np.isfinite(nominal_duration_s):
        nominal_duration_s = float(stim["stim_duration_s"].median())

    stim["nominal_duration_s"] = float(nominal_duration_s)

    low = nominal_duration_s * (1 - DURATION_TOLERANCE)
    high = nominal_duration_s * (1 + DURATION_TOLERANCE)
    stim["duration_ok"] = stim["stim_duration_s"].between(low, high, inclusive="both")

    if "pulse_count" in stim.columns:
        stim["measured_frequency_hz"] = measured_frequency(stim["pulse_count"], stim["stim_duration_s"])
    elif "actual_frequency_hz" in stim.columns:
        stim["measured_frequency_hz"] = pd.to_numeric(stim["actual_frequency_hz"], errors="coerce")
    else:
        raise ValueError("stim table has neither pulse_count nor actual_frequency_hz")

    nominal, error = classify_frequency(stim["measured_frequency_hz"])
    stim["nominal_frequency_hz"] = nominal
    stim["frequency_error"] = error
    stim["frequency_ok"] = nominal.notna()

    stim["is_good_stim"] = stim["duration_ok"] & stim["frequency_ok"]

    stim["reject_reason"] = np.select(
        [
            stim["is_good_stim"],
            ~stim["duration_ok"] & ~stim["frequency_ok"],
            ~stim["duration_ok"],
            ~stim["frequency_ok"],
        ],
        ["", "duration_and_frequency", "duration", "frequency"],
        default="",
    )

    return stim


def summarise_rejections(stim: pd.DataFrame) -> pd.DataFrame:
    """One row per reject reason, for the segmentation report."""
    counts = stim["reject_reason"].replace("", "accepted").value_counts()
    return counts.rename_axis("verdict").reset_index(name="n_stimuli")


def cells_below_minimum(trial_counts: pd.DataFrame, n_col: str = "n_trials") -> pd.DataFrame:
    """Rows of a per-cell table that do not clear MIN_TRIALS_PER_CELL."""
    return trial_counts[trial_counts[n_col] < MIN_TRIALS_PER_CELL]
