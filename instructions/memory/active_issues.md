# Active Issues

Issues that are currently broken or blocking further work, in priority order.

---

## BLOCKING

### NB07 — Clinical Prediction needs factor scores
- **File:** `notebooks/03_vigor_analysis/07_clinical_prediction.ipynb`
- **Issue:** Loads `results/stats/modeling_factor_param.csv` which does not exist.
- **What's needed:** Factor analysis (EFA/CFA) of the psychiatric battery (DASS-21, PHQ-9, OASIS, STAI, AMI, MFIS, STICSA) to produce factor scores per subject.
- **Inputs available:** `data/exploratory_350/processed/stage5_filtered_data_*/psych.csv` (N=293, all subscales scored)
- **Resolution:** Run EFA, save output to `results/stats/modeling_factor_param.csv` with column `subj`.

---

## TECHNICAL DEBT

### parameter_recovery.ipynb not yet run
- **File:** `notebooks/02_choice_modeling/02_parameter_recovery.ipynb`
- **Issue:** Has not been executed against the new N=293 fit.
- **Resolution:** Run after confirming fit quality.

### Full model comparison not run on N=293
- **Issue:** Only FETExponentialBias was fitted on the full N=293 dataset. The WAIC comparison table is from the old 270-subject GPU fit.
- **Resolution:** Run all 7 models (FETExponential, FETHyperbolic, FETLinear, FETQuadratic, FETExponentialBias, ThreatOnly, EffortOnly) on N=293 for the paper.

### Confirmatory sample (N=350) not started
- **Issue:** Raw data exists but preprocessing pipeline has not been run on confirmatory sample.
- **Resolution:** Run full pipeline (stage1→stage5), then refit FETExponentialBias, rerun all vigor analyses.

### Draft main.md needs updating
- **Issue:** Draft contains placeholder affect values. Real values are now computed (NB12 complete).
- **Core results:** anxiety β=+0.575 (p_threat), β=−0.602 (S_probe); confidence β=−0.586 (p_threat), β=+0.632 (S_probe)
- **State-trait:** z → trait confidence (β=−0.719, p=0.044); κ → trait anxiety (+) and confidence (−); z×threat moderation NULL
- **Cross-domain null:** vigor × affect correlations all n.s. (FDR-corrected) — parallel but independent reactive systems
- **Resolution:** Update draft with two-system framing and actual results from `results/stats/affect_lmm_results.csv`.

---

## INFRASTRUCTURE

### Disk space is tight
- 259 GB free after cleanup (was 2.9 GB).
- `Desktop/Lima-Analysis/` still has 94 GB of old fits — candidate for archival/deletion.
- `data/exploratory_350/processed/stage2_trial_processing_*/processed_trials.pkl` is 1.5 GB; will re-accumulate if pipeline is re-run.
