# Alpha anesthesia — 40 Hz vs 100 Hz visual stimulation

Analysis pipeline for mouse recordings under isoflurane anesthesia. Each recording delivers
trains of flickering light at a nominal 40 Hz or 100 Hz while EEG, pupil diameter, heart rate
and respiration are recorded. The pipeline cuts every stimulus into a window, normalises each
channel to its own pre-stimulus baseline, and aggregates across trials, mice and conditions.

## Dataset

- 4 mice (`NGexp9`, `NGexp10`, `NGexp12`, `NGexp13`), 15 recordings, 29 Jun – 6 Jul 2026.
- Two anesthesia depths per mouse: `light` and `deep` (isoflurane ~0.8–1.6%).
- Metadata: `data/alpha_recordings.csv`, `data/alpha_recordings - july.csv`.

Raw and segmented data are **not** in this repo — they are ~10 GB and fully regenerable.
They live in the shared Google Drive folder `Ally's project`.

## Pipeline

| Script | Role |
|---|---|
| `segment_alpha_recording.py` | Core library. Cuts −10 s → +60 s windows around each stimulus onset, normalises to the median of the 10 s pre-stimulus baseline, keeps only stimuli lasting 0.8–1.1 s, writes per-recording CSVs and plots. Also runnable as a CLI. |
| `main.py` | Single recording, hard-coded file paths. |
| `run_segmentation_simple.py` | Minimal single-recording entry point. |
| `batch_run_all.py` | Iterates every row of `alpha_recordings - july.csv`, locates each recording folder, runs the segmentation, and builds `combined_normalized_trials/`. |
| `group_plots.py` | Mean ± SD across mice per metric and anesthesia depth, 40 Hz vs 100 Hz. |
| `summary_bar_plots.py` | Bar plots of the mean change in the 0–10 s window after stimulus onset. |

## Metrics

`eeg_rel_delta`, `eeg_supp_mask_probability`, `pupil_diameter_mm`, `heart_rate_hr_bpm`,
`respiration_resp_bpm` — all reported as percent change from each trial's own baseline,
except the suppression probability which is a raw 0–1 value.

## Headline result (mean across 4 mice, 0–10 s post-stimulus)

| Metric | light 40 Hz | light 100 Hz | deep 40 Hz | deep 100 Hz |
|---|---|---|---|---|
| EEG rel_delta | −23.2% | −10.5% | −29.2% | −20.6% |
| Pupil diameter | +57.9% | +32.3% | +23.3% | +29.6% |
| Respiration rate | +12.5% | +10.2% | +36.4% | +28.6% |
| Heart rate | +1.2% | +0.6% | +2.3% | +1.1% |

40 Hz produces the stronger arousal signature in most conditions. **n = 4 mice and the
between-mouse SD is large** (e.g. deep 40 Hz pupil: mean 23.3%, SD 18.2%, one mouse negative).
No statistical testing is implemented yet.

Numbers: `results/summary_bar_plots/summary_0_10s_group.csv` and `…_per_mouse.csv`.

## Running it

```bash
pip install pandas numpy matplotlib
python batch_run_all.py     # expects ally_pupil/ and ally_results/ beside the scripts
python group_plots.py
python summary_bar_plots.py
```

## Known rough edges

- `main.py` and `run_segmentation_simple.py` carry hard-coded filenames for one recording.
- `batch_run_all.py` assumes folder names of the form `<mouse>-…-<filename>` and raises if the match is not unique.
- No error handling around missing modalities beyond a printed skip.
- No statistics (no per-condition test, no correction for multiple comparisons).
