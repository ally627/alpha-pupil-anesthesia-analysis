"""Rebuild `combined_normalized_trials/` from existing `normalized_trials/` folders.

`batch_run_all.py` builds the combined tables from the full segmented CSVs, which means
it needs the raw recordings. When only the per-recording `normalized_trials/` tables are
at hand — they are three orders of magnitude smaller — this script reaches the same
result: pool every trial of one (mouse, anesthesia, frequency, metric) across recordings,
recompute mean and SD, and drop cells with too few good trials.

Usage:
    python rebuild_aggregates.py --source <folder containing the segmented_output_* dirs>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from quality_control import MIN_TRIALS_PER_CELL


FILENAME_PATTERN = re.compile(
    r"^(?P<mouse>.+?)-(?P<anes>.+?)-(?P<freq>\d+)Hz-(?P<metric>.+)-n(?P<n_trials>\d+)\.csv$"
)


def parse_name(path: Path):
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        return None
    info = match.groupdict()
    info["freq"] = int(info["freq"])
    info["n_trials"] = int(info["n_trials"])
    return info


def rebuild(source: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    cells: dict[tuple, list[pd.DataFrame]] = {}

    for path in sorted(source.glob("**/normalized_trials/*.csv")):
        info = parse_name(path)
        if info is None:
            print(f"Skipping unexpected filename: {path.name}")
            continue

        table = pd.read_csv(path)
        if "time_from_stim" not in table.columns:
            print(f"Skipping {path.name}: no time_from_stim column")
            continue

        # Drop the per-recording aggregates; they are recomputed over the pooled trials.
        trial_cols = [c for c in table.columns if c not in ("time_from_stim", "mean", "std")]

        # Trial ids repeat across recordings (stim_00, stim_01 …), so qualify them.
        recording = path.parent.parent.name
        table = table[["time_from_stim"] + trial_cols].rename(
            columns={c: f"{recording}::{c}" for c in trial_cols}
        )

        key = (info["mouse"], info["anes"], info["freq"], info["metric"])
        cells.setdefault(key, []).append(table.set_index("time_from_stim"))

    written = []

    for (mouse, anes, freq, metric), tables in sorted(cells.items()):
        pooled = pd.concat(tables, axis=1).sort_index()
        n_trials = pooled.shape[1]

        if n_trials < MIN_TRIALS_PER_CELL:
            print(
                f"Excluding {mouse} {anes} {freq}Hz {metric}: "
                f"{n_trials} good trial(s) across {len(tables)} recording(s), "
                f"minimum is {MIN_TRIALS_PER_CELL}"
            )
            continue

        pooled["mean"] = pooled.mean(axis=1)
        pooled["std"] = pooled.drop(columns=["mean"]).std(axis=1)

        out_name = f"{mouse}-{anes}-{freq}Hz-{metric}-n{n_trials}.csv"
        pooled.reset_index().to_csv(output_dir / out_name, index=False)
        written.append({"mouse": mouse, "anes": anes, "frequency_hz": freq,
                        "metric": metric, "n_trials": n_trials,
                        "n_recordings": len(tables)})
        print(f"Saved combined: {out_name}")

    return pd.DataFrame(written)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."),
                        help="Folder to search recursively for normalized_trials/ tables")
    parser.add_argument("--output-dir", type=Path, default=Path("combined_normalized_trials"))
    args = parser.parse_args()

    written = rebuild(args.source, args.output_dir)

    if written.empty:
        print("Nothing rebuilt.")
        return

    inventory = args.output_dir / "cell_inventory.csv"
    written.to_csv(inventory, index=False)
    print(f"\n{len(written)} cells written. Inventory: {inventory}")


if __name__ == "__main__":
    main()
