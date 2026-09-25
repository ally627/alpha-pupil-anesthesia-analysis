# Pupil and physiological responses to nociceptive stimulation under isoflurane anesthesia

Analysis code for Ally Taguri's research project. Mice under isoflurane anesthesia at two
depths (light 0.8–1.2 %, deep 1.3–1.6 %) received 1 s trains of nociceptive tail stimulation
at 40 Hz or 100 Hz while pupil diameter, EEG, heart rate and respiration were recorded.
The code cuts every stimulus into a window, expresses each signal relative to its own
pre-stimulus baseline, aggregates across trials and mice, and tests the effects of
stimulation frequency and anesthesia depth.

## Dataset

- 4 mice (`NGexp9`, `NGexp10`, `NGexp12`, `NGexp13`), 2 recordings per mouse per anesthesia
  depth, except NGexp12 deep (1 recording): **15 recordings**, 12 stimuli each (6 per frequency).
- Metadata: `data/alpha_recordings - july.csv`.
- Raw and intermediate data (~10 GB) are **not** in this repository; they are kept in Google
  Drive and every derived file here can be regenerated from them.

## Analysis steps

Run in this order. Every script takes `--help`.

| Step | Script | What it does |
|---|---|---|
| 1 | `build_curves.py` | Reads the raw files, applies stimulus quality control, cuts −20 s → +60 s around every stimulus onset and baseline-corrects each trial against the 10 s before onset. Output: one long table of per-trial curves. |
| 2 | `noise_screening.py` | Screens the raw EEG of every trial for noise, saturation and flat segments (a check; no trials were excluded by it). |
| 3 | `aggregate.py` | Median across trials within each mouse, then mean ± SEM across mice. Group and per-mouse curve plots. |
| 4 | `features.py` | Reduces each trial to summary numbers (mean over 0–10 s, peak, latency) and fits the statistical model. |
| 5 | `simple_effects.py` | The 40 vs 100 Hz effect within each anesthesia depth separately. |
| 6 | `screen_outliers.py` | Flags trials that differ strongly from the other trials of the same mouse and condition (a check; no trials were excluded by it). |
| 7 | `summary_figures.py`, `cross_metric_figure.py` | The figures in the thesis, including the comparison of all metrics on one scale. |

Shared definitions: `quality_control.py` (inclusion rules), `derived_metrics.py` (metrics and
their units), `segment_alpha_recording.py` (the original segmentation code; its stimulus-table
reader is still used by steps 1 and 2).

## Methods in brief

**Stimulus quality control.** A stimulus is kept only if its duration and its delivered pulse
rate (pulses ÷ duration) are close to the nominal values. 55 of 180 stimuli failed. A mouse
contributes to a condition only with at least 3 good stimuli; this removed one more
(NGexp12, deep, 40 Hz). **124 stimuli** entered the analysis.

**Units.**
- Pupil diameter, heart rate, respiration rate: percent change from baseline.
- Heart rate variability (SDNN: standard deviation of beat-to-beat intervals in a sliding
  10 s window): change from baseline in **ms**, because its baseline is small enough that a
  percent change inflates ordinary responses.
- EEG relative delta power (already a fraction of total power): change in **percentage points**.

**Stimulus artefact.** The stimulator is picked up by the EEG electrodes during the train
(a spectral peak at the delivered pulse rate in all 15 recordings), which inflates total power
and artificially lowers relative delta. EEG metrics therefore exclude the first 1.5 s after
onset. Pupil, heart rate and respiration are unaffected.

**Aggregation.** Median across a mouse's stimuli in each condition, then mean ± SEM across mice.

**Statistics.** Per-trial mean change over 0–10 s, modelled with a linear mixed model
`y ~ frequency * depth + (1 | mouse)` (statsmodels), followed by the frequency effect within
each depth. Pupil diameter was defined in advance as the primary outcome and is reported
without correction; the other metrics are corrected for multiple comparisons with
Benjamini–Hochberg. α = 0.05.

## Running

```bash
pip install pandas numpy scipy matplotlib statsmodels
python build_curves.py --raw <raw data folder> --output curves_per_trial.csv
python noise_screening.py --raw <raw data folder>
python aggregate.py
python features.py
python simple_effects.py
python screen_outliers.py
python summary_figures.py
python cross_metric_figure.py
```

Outputs are written to `results/`.

## Older scripts

`batch_run_all.py`, `main.py`, `run_segmentation_simple.py`, `rebuild_aggregates.py`,
`group_plots.py`, `summary_bar_plots.py` and `combined_pupil_figure.py` are from the first
version of the pipeline. They established the design the current steps follow (windows around
each stimulus, baseline correction, the 0–10 s response window, 40 vs 100 Hz within each
depth) but were not run to produce the results in the thesis.
