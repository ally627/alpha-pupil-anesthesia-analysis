"""Frequency effect within each anesthesia depth (simple effects).

The interaction term in features.py says the 40-vs-100 Hz difference is not the
same at both depths, but not where it holds. This refits the same linear mixed
model, y ~ C(freq) * C(anes) + (1 | mouse), once with each depth as the
reference level, and reports the frequency coefficient at that depth.

Same inputs and rules as features.py: per-trial features, MIN_TRIALS_PER_CELL
applied per mouse and condition, response z-scored for numerical stability and
the coefficient rescaled back to the metric's units.

Caveat worth stating in the write-up: the model has a random intercept per
mouse but no random slope for frequency, so mouse-to-mouse disagreement about
the frequency effect is not modelled and the p-value is optimistic. With four
mice a per-mouse paired test cannot reach p < 0.125 at all.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

MIN_TRIALS_PER_CELL = 3
OPTIMISERS = ("powell", "nm", "bfgs", "lbfgs")
DEFAULT_METRICS = ("pupil_diameter_mm", "respiration_resp_bpm")
DEFAULT_FEATURES = ("mean_0_10",)


def fit_at(block: pd.DataFrame, reference: str):
    formula = ("z ~ C(nominal_frequency_hz, Treatment(40))"
               f" * C(anes, Treatment('{reference}'))")
    last_error = None
    for method in OPTIMISERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return smf.mixedlm(formula, block, groups=block["mouse"]).fit(
                    method=method, reml=True)
        except Exception as error:  # singular fits: try the next optimiser
            last_error = error
    raise RuntimeError(f"all optimisers failed: {last_error}")


def simple_effects(trials: pd.DataFrame, metrics, features) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        for feature in features:
            block = trials[trials["metric"] == metric].dropna(subset=[feature]).copy()
            counts = block.groupby(["mouse", "anes", "nominal_frequency_hz"])["trial_id"]
            block = block[counts.transform("nunique") >= MIN_TRIALS_PER_CELL]
            spread = block[feature].std()
            block["z"] = (block[feature] - block[feature].mean()) / spread

            for depth in ("light", "deep"):
                result = fit_at(block, depth)
                term = next(name for name in result.params.index
                            if name.startswith("C(nominal") and ":" not in name)
                rows.append({
                    "metric": metric,
                    "feature": feature,
                    "anes": depth,
                    "effect_100_minus_40": result.params[term] * spread,
                    "p": result.pvalues[term],
                    "n_trials": int((block["anes"] == depth).sum()),
                    "n_mice": block.loc[block["anes"] == depth, "mouse"].nunique(),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("results/features_per_trial.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/simple_effects.csv"))
    args = parser.parse_args()

    trials = pd.read_csv(args.features)
    table = simple_effects(trials, DEFAULT_METRICS, DEFAULT_FEATURES)
    table.to_csv(args.output, index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
