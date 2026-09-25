"""Reduce every trial curve to a few numbers, then test them.

A curve is not a result. To claim "40 Hz drives a larger response than 100 Hz" you need one
number per trial, and a model that knows those numbers came from four mice rather than from
sixty independent animals. This module does both (issues #6, #19).

Features, all computed on the baseline-corrected curve:

* `peak`              largest deflection in 0-30 s, signed by the direction of the response
* `latency_to_peak_s` when that peak occurred
* `mean_0_10`         mean over 0-10 s, the early response window
* `auc_0_10`          area under the curve over 0-10 s (units x seconds)
* `auc_0_30`          area under the curve over 0-30 s
* `return_s`          first time after the peak at which the curve falls back below 25 %
                      of it, i.e. how long the response lasted; NaN if it never does

Statistics: a linear mixed-effects model per metric and feature,

    feature ~ C(frequency) * C(anes)  +  (1 | mouse)

The random intercept for `mouse` is the point. Each mouse contributes several trials in
each condition, and trials from one animal resemble each other; an ordinary ANOVA treats
them as independent and inflates significance. The mixed model estimates a separate
baseline per mouse, removes it, and tests the fixed effects on what is left. It also
tolerates unequal trial counts per cell, which is exactly what quality control produced.

The interaction term is the scientifically interesting one: it asks whether the 40 Hz
advantage itself changes between light and deep anesthesia.

Alongside the model, a paired Wilcoxon signed-rank test on per-mouse medians is reported.
It is weak with n=4 and cannot be significant below n=6, but it is assumption-free and
serves as a sanity check on the model's direction.

p-values are corrected across metrics within a feature by Benjamini-Hochberg FDR.

Usage:
    python features.py --curves curves_per_trial.csv --output-dir results
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

from derived_metrics import METRIC_BY_KEY
from quality_control import MIN_TRIALS_PER_CELL


# The stimulator leaks into the EEG electrodes during the ~1 s train. Checked on all 15
# recordings (raw 1000 Hz trace, stimulus window vs the preceding 10 s): every recording
# shows a spectral peak at the exact delivered pulse rate (40 or 100 Hz), median 2x-5000x
# the neighbouring bins. Strength varies by mouse: NGexp9 is gross (amplitude 2-14x
# baseline, one recording saturated for ~half the train); the other three mice show the
# peak with little amplitude change. Extra power at 40/100 Hz shrinks *relative* delta
# even if the brain does nothing, so EEG metrics are blanked over the train plus 0.5 s.
# Pupil is optical; HR and respiration come from the monitor; neither is blanked.
ARTEFACT_BLANK_S = 1.5
ARTEFACT_METRICS = ("eeg_rel_delta", "eeg_bsr")

PEAK_WINDOW_S = (0.0, 30.0)
EARLY_WINDOW_S = (0.0, 10.0)
LATE_WINDOW_S = (0.0, 30.0)
RETURN_FRACTION = 0.25


def trial_features(curve: pd.DataFrame) -> dict[str, float]:
    """One trial in, one row of numbers out. `curve` is sorted, baseline-corrected."""
    time = curve["time_from_stim"].to_numpy()
    value = curve["value"].to_numpy()

    window = (time >= PEAK_WINDOW_S[0]) & (time <= PEAK_WINDOW_S[1])
    if window.sum() < 3 or not np.isfinite(value[window]).any():
        return {}

    t_win, v_win = time[window], value[window]

    # Signed peak: the largest excursion in either direction, keeping its sign, so that a
    # metric that drops (EEG delta) and one that rises (pupil) are both described honestly.
    finite = np.isfinite(v_win)
    idx = np.nanargmax(np.abs(v_win[finite]))
    peak = float(v_win[finite][idx])
    latency = float(t_win[finite][idx])

    def mean_over(lo: float, hi: float) -> float:
        mask = (time >= lo) & (time <= hi) & np.isfinite(value)
        return float(np.mean(value[mask])) if mask.any() else np.nan

    def auc_over(lo: float, hi: float) -> float:
        mask = (time >= lo) & (time <= hi) & np.isfinite(value)
        return float(np.trapezoid(value[mask], time[mask])) if mask.sum() > 2 else np.nan

    after_peak = (time > latency) & np.isfinite(value)
    threshold = abs(peak) * RETURN_FRACTION
    fell_back = after_peak & (np.abs(value) < threshold)
    return_s = float(time[fell_back][0] - latency) if fell_back.any() else np.nan

    return {
        "peak": peak,
        "latency_to_peak_s": latency,
        "mean_0_10": mean_over(*EARLY_WINDOW_S),
        "auc_0_10": auc_over(*EARLY_WINDOW_S),
        "auc_0_30": auc_over(*LATE_WINDOW_S),
        "return_s": return_s,
    }


def build_features(curves: pd.DataFrame) -> pd.DataFrame:
    # Same inclusion rule as the curves: a cell needs MIN_TRIALS_PER_CELL good trials or it
    # is not analysed at all. Without this the statistics would silently include a mouse
    # represented by a single trial while the figures excluded it, and the two would
    # disagree about n.
    counts = curves.groupby(["mouse", "anes", "nominal_frequency_hz", "metric"])["trial_id"]
    curves = curves[counts.transform("nunique") >= MIN_TRIALS_PER_CELL]

    rows = []

    keys = ["recording", "mouse", "anes", "trial_id", "nominal_frequency_hz", "metric"]
    for key_values, curve in curves.sort_values("time_from_stim").groupby(keys):
        metric = key_values[keys.index("metric")]
        if metric in ARTEFACT_METRICS:
            curve = curve[(curve["time_from_stim"] < 0) |
                          (curve["time_from_stim"] >= ARTEFACT_BLANK_S)]

        computed = trial_features(curve)
        if not computed:
            continue
        rows.append(dict(zip(keys, key_values)) | computed)

    return pd.DataFrame(rows)


FEATURES = ["peak", "latency_to_peak_s", "mean_0_10", "auc_0_10", "auc_0_30", "return_s"]


def fit_mixed_model(data: pd.DataFrame, feature: str) -> dict[str, float] | None:
    """feature ~ frequency * anes with a random intercept per mouse."""
    work = data[["mouse", "anes", "nominal_frequency_hz", feature]].dropna().copy()
    work = work.rename(columns={feature: "y", "nominal_frequency_hz": "freq"})
    work["freq"] = work["freq"].astype(int).astype(str)

    if work["mouse"].nunique() < 3 or len(work) < 12:
        return None
    if work["freq"].nunique() < 2 or work["anes"].nunique() < 2:
        return None

    # The features live on wildly different scales — AUC in the thousands, BSR in
    # hundredths. The likelihood optimiser goes singular on the large ones, so fit on a
    # z-scored response and multiply the coefficients back afterwards. Same model, same
    # p-values, numerically stable.
    scale = work["y"].std(ddof=1)
    if not np.isfinite(scale) or scale == 0:
        return None
    work["yz"] = (work["y"] - work["y"].mean()) / scale

    fit = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm("yz ~ C(freq) * C(anes)", work, groups=work["mouse"])
        for method in ("powell", "nm", "bfgs", "lbfgs"):
            try:
                fit = model.fit(reml=True, method=method)
                break
            except Exception as error:
                last_error = str(error)[:80]

    if fit is None:
        return {"error": last_error}

    def term(name: str) -> tuple[float, float]:
        match = [p for p in fit.params.index if name in p]
        if not match:
            return np.nan, np.nan
        return float(fit.params[match[0]]) * scale, float(fit.pvalues[match[0]])

    freq_beta, freq_p = term("C(freq)[T.40]")
    if not np.isfinite(freq_beta):
        freq_beta, freq_p = term("C(freq)[T.100]")
        freq_beta = -freq_beta  # report everything as the 40 Hz effect

    anes_beta, anes_p = term("C(anes)[T.light]")
    inter_beta, inter_p = term(":")

    return {
        "n_obs": len(work),
        "n_mice": work["mouse"].nunique(),
        "freq_effect_40_vs_100": freq_beta,
        "freq_p": freq_p,
        "anes_effect": anes_beta,
        "anes_p": anes_p,
        "interaction": inter_beta,
        "interaction_p": inter_p,
    }


def paired_test(data: pd.DataFrame, feature: str) -> dict[str, float]:
    """Wilcoxon on per-mouse medians, 40 vs 100, within each anesthesia depth."""
    per_mouse = (
        data.groupby(["mouse", "anes", "nominal_frequency_hz"])[feature]
        .median()
        .unstack("nominal_frequency_hz")
    )
    if 40 not in per_mouse.columns or 100 not in per_mouse.columns:
        return {}

    results = {}
    for anes, block in per_mouse.groupby("anes"):
        pairs = block[[40, 100]].dropna()
        if len(pairs) < 3:
            continue
        difference = pairs[40] - pairs[100]
        try:
            _, p = stats.wilcoxon(pairs[40], pairs[100])
        except ValueError:
            p = np.nan
        results[f"{anes}_median_diff"] = float(difference.median())
        results[f"{anes}_n_mice"] = int(len(pairs))
        results[f"{anes}_wilcoxon_p"] = float(p)
        # Cohen's dz on the paired differences: the effect size the p-value cannot show.
        if difference.std(ddof=1) > 0:
            results[f"{anes}_dz"] = float(difference.mean() / difference.std(ddof=1))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=Path("curves_per_trial.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    curves = pd.read_csv(args.curves)
    features = build_features(curves)
    features.to_csv(args.output_dir / "features_per_trial.csv", index=False)
    print(f"{len(features):,} trial-metric rows of features")

    per_mouse = (
        features.groupby(["mouse", "anes", "nominal_frequency_hz", "metric"])[FEATURES]
        .median()
        .reset_index()
    )
    per_mouse.to_csv(args.output_dir / "features_per_mouse.csv", index=False)

    rows = []
    for metric, block in features.groupby("metric"):
        for feature in FEATURES:
            model = fit_mixed_model(block, feature)
            if model is None:
                continue
            rows.append({"metric": metric, "feature": feature} | model | paired_test(block, feature))

    stats_table = pd.DataFrame(rows)

    # Multiple comparisons: one family per feature, across the seven metrics.
    for column in ["freq_p", "interaction_p", "light_wilcoxon_p", "deep_wilcoxon_p"]:
        if column not in stats_table.columns:
            continue
        stats_table[column + "_fdr"] = np.nan
        for feature, block in stats_table.groupby("feature"):
            valid = block[column].notna()
            if valid.sum() < 2:
                continue
            corrected = multipletests(block.loc[valid, column], method="fdr_bh")[1]
            stats_table.loc[block.index[valid], column + "_fdr"] = corrected

    stats_table.to_csv(args.output_dir / "statistics.csv", index=False)

    print("\nMixed model, 40 Hz vs 100 Hz effect (positive = 40 Hz larger):")
    show = stats_table[stats_table["feature"].isin(["mean_0_10", "peak", "auc_0_30"])]
    columns = ["metric", "feature", "freq_effect_40_vs_100", "freq_p", "freq_p_fdr",
               "interaction", "interaction_p", "n_obs", "n_mice"]
    print(show[[c for c in columns if c in show.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
