---
result_id: 301
class: vigor_dynamics
title: Vigor processing pipeline — kernel smoothing, phase metrics, condition-cell means
status: supported
prereg_h: []
internal_h: []
samples: [exploratory_290, confirmatory_281]
notebooks: [notebooks/analysis/load_data.py]
scripts: []
outputs: [data/exploratory_350/processed/vigor_processed/, data/confirmatory_350/processed/vigor_processed/, results/stats/vigor_analysis/cell_means_exploratory.csv]
figures: [results/stats/vigor_analysis/fig_encounter_timecourse.png]
created: 2026-05-28
last_run: 2026-05-28
---

# Result 301 — Vigor processing pipeline (methods)

## Overview

This is a methods-only result documenting the pipeline that produces all vigor-derived metrics used in the project. Raw keypress timestamps are converted to instantaneous rates via inter-press intervals, smoothed with a Gaussian kernel at 20 Hz, and phase-aligned to trial events (onset, encounter, terminal). Per-subject condition cell means (subject × threat × distance × cookie) are the unit for the joint-model vigor likelihood.

## Method

1. **Raw inputs:** keypress timestamps per trial from Unity logs.
2. **IPI filtering:** discard intervals < 10 ms as artifacts.
3. **Instantaneous rate:** `1 / IPI` at each timestamp.
4. **Normalization:** divide by per-subject, per-block `calibrationMax`.
5. **Smoothing:** Gaussian kernel (σ ≈ 100 ms) at 20 Hz, producing `smoothed_vigor_ts.parquet`.
6. **Phase extraction:** segment timecourse into pre-encounter, encounter, terminal phases relative to `encounterTime`.
7. **Aggregation:** per-trial phase metrics in `phase_trial_metrics.parquet`; per-subject summary in `subject_vigor_table.csv`; per-cell means in `cell_means_{sample}.csv`.

## Outputs

- `data/{sample}/processed/vigor_processed/smoothed_vigor_ts.parquet` — 20 Hz smoothed timecourse, all subjects × trials.
- `data/{sample}/processed/vigor_processed/phase_trial_metrics.parquet` — per-trial × phase metrics.
- `data/{sample}/processed/vigor_processed/subject_vigor_table.csv` — per-subject summaries.
- `results/stats/vigor_analysis/cell_means_{sample}.csv` — per-(subject, threat, distance, cookie) cell means used as the joint-model vigor likelihood input.

## Replication

Vigor pipeline is invoked via `from load_data import load_both` (in `notebooks/analysis/load_data.py`). Smoothing and phase extraction are implemented as functions called during data load; outputs are cached on disk and not regenerated unless raw data changes.

## References

- [[result_103]] — H1c uses these cell means.
- [[result_201]] — Joint model vigor likelihood uses these cell means.
- All 300-block results depend on this pipeline.
