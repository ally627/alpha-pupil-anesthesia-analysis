"""Look at the raw EEG before trusting anything computed from it.

Every downstream EEG number — relative delta, burst-suppression ratio — is a transformation
of the raw trace. If the electrode was noisy for part of a recording, those numbers are
still produced, still smooth, and still wrong. This module screens the raw signal itself
(issue #18).

Per trial window it computes:

* `amplitude_sd`    standard deviation of the raw trace
* `hf_index`        SD of the sample-to-sample difference divided by `amplitude_sd`.
                    Physiological EEG is dominated by slow components under anesthesia,
                    so this ratio is low; electrical or movement noise raises it.
* `saturated_frac`  fraction of samples at or beyond 99.5 % of the recording's own range,
                    i.e. clipping
* `flat_frac`       fraction of samples where the trace does not move at all, which means
                    a disconnected or dead channel

Trials are compared against the other trials of the same recording, so a chronically
noisy animal is judged against itself.

Output is a CSV of candidates plus one PNG per recording showing the worst trial against a
typical one, because a plot settles this question faster than any threshold.

Usage:
    python noise_screening.py --raw raw/ally_results --output-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_curves import PRE_SECONDS, POST_SECONDS, locate
from segment_alpha_recording import prepare_stim_table


NOISE_Z = 3.5
SATURATION_LIMIT = 0.02
FLAT_LIMIT = 0.05


def screen_recording(raw_root: Path, recording: str, mouse: str, anes: str,
                     nominal_duration_s: float | None) -> tuple[pd.DataFrame, dict]:
    paths = locate(raw_root, recording)
    if paths["stim"] is None or paths["eeg"] is None:
        return pd.DataFrame(), {}

    stim = prepare_stim_table(pd.read_csv(paths["stim"]), nominal_duration_s=nominal_duration_s)
    eeg = pd.read_csv(paths["eeg"], usecols=lambda c: c in {"time", "raw"})
    eeg = eeg.sort_values("time")

    times = eeg["time"].to_numpy()
    values = eeg["raw"].to_numpy()

    low, high = np.nanpercentile(values, [0.25, 99.75])
    span = high - low

    rows = []
    traces = {}

    for _, stimulus in stim.iterrows():
        if not bool(stimulus["is_good_stim"]):
            continue

        onset = float(stimulus["stim_time"])
        start, end = np.searchsorted(times, [onset - PRE_SECONDS, onset + POST_SECONDS])
        segment = values[start:end]
        if len(segment) < 100:
            continue

        finite = segment[np.isfinite(segment)]
        if len(finite) < 100:
            continue

        differences = np.diff(finite)
        amplitude_sd = float(np.std(finite))

        rows.append({
            "recording": recording, "mouse": mouse, "anes": anes,
            "trial_id": stimulus["trial_id"],
            "nominal_frequency_hz": int(stimulus["nominal_frequency_hz"]),
            "amplitude_sd": amplitude_sd,
            "hf_index": float(np.std(differences) / amplitude_sd) if amplitude_sd > 0 else np.nan,
            "saturated_frac": float(np.mean((finite <= low) | (finite >= high))),
            "flat_frac": float(np.mean(differences == 0)),
        })
        traces[stimulus["trial_id"]] = (times[start:end] - onset, segment)

    return pd.DataFrame(rows), traces


def plot_extremes(traces: dict, screened: pd.DataFrame, recording: str, output_dir: Path) -> None:
    if screened.empty or not traces:
        return

    worst = screened.sort_values("hf_index", ascending=False).iloc[0]
    typical = screened.iloc[(screened["hf_index"] -
                             screened["hf_index"].median()).abs().argsort()].iloc[0]

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, sharey=True)
    for ax, row, label in zip(axes, [typical, worst], ["typical trial", "noisiest trial"]):
        time, value = traces[row["trial_id"]]
        ax.plot(time, value, lw=0.4, color="#2e86ab" if label.startswith("typical") else "#d1495b")
        ax.axvline(0, lw=0.8, color="k", alpha=0.6)
        ax.set_ylabel("Raw EEG")
        ax.set_title(f"{label}: {row['trial_id']} — "
                     f"HF index {row['hf_index']:.2f}, SD {row['amplitude_sd']:.1f}", fontsize=10)
        ax.grid(alpha=0.12)

    axes[-1].set_xlabel("Time from stimulus onset (s)")
    fig.suptitle(recording, fontsize=10)
    fig.tight_layout()

    plots = output_dir / "noise_screening"
    plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots / f"{recording}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("raw/ally_results"))
    parser.add_argument("--recordings", type=Path, default=Path("data/alpha_recordings - july.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    alpha = pd.read_csv(args.recordings)
    folders = {p.name for p in args.raw.iterdir() if p.is_dir() and p.name != "ally_pupil"}

    all_rows = []

    for _, row in alpha.iterrows():
        mouse, filename = str(row["mouse"]).strip(), str(row["filename"]).strip()
        matches = [f for f in folders if f.startswith(f"{mouse}-") and f.endswith(filename)]
        if len(matches) != 1:
            continue

        screened, traces = screen_recording(
            args.raw, matches[0], mouse, str(row["anes"]).strip(),
            pd.to_numeric(row.get("duration_s"), errors="coerce"))

        if screened.empty:
            continue

        plot_extremes(traces, screened, matches[0], args.output_dir)
        all_rows.append(screened)
        print(f"{matches[0]}: {len(screened)} trials screened")

    if not all_rows:
        print("Nothing screened.")
        return

    screened = pd.concat(all_rows, ignore_index=True)

    for column in ["amplitude_sd", "hf_index"]:
        screened[f"z_{column}"] = screened.groupby("recording")[column].transform(
            lambda s: 0.6745 * (s - s.median()) / max((s - s.median()).abs().median(), 1e-12)
        )

    def verdict(row: pd.Series) -> str:
        if row["flat_frac"] > FLAT_LIMIT:
            return "exclude"
        if row["saturated_frac"] > SATURATION_LIMIT:
            return "inspect"
        if abs(row["z_hf_index"]) > NOISE_Z or abs(row["z_amplitude_sd"]) > NOISE_Z:
            return "inspect"
        return "ok"

    screened["verdict"] = screened.apply(verdict, axis=1)
    screened.to_csv(args.output_dir / "eeg_noise_screen.csv", index=False)

    counts = screened["verdict"].value_counts()
    print(f"\n{len(screened)} trials: {counts.get('ok', 0)} ok, "
          f"{counts.get('inspect', 0)} to inspect, {counts.get('exclude', 0)} to exclude")

    flagged = screened[screened["verdict"] != "ok"]
    for _, row in flagged.iterrows():
        print(f"  [{row['verdict']}] {row['mouse']} {row['anes']} {row['trial_id']}: "
              f"HF z={row['z_hf_index']:+.1f}, amplitude z={row['z_amplitude_sd']:+.1f}, "
              f"saturated {row['saturated_frac']:.1%}, flat {row['flat_frac']:.1%}")


if __name__ == "__main__":
    main()
