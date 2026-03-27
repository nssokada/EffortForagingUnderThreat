"""
PPC + parameter extraction for the fitted FETExponentialBias model.
Saves results to results/stats/.
"""

import sys
from pathlib import Path
import glob
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from modeling import ModelFitter, load_fitted_models
from modeling.ppc import PosteriorPredictive, compute_waic

SAVE_DIR  = ROOT / 'results' / 'model_fits' / 'exploratory'
STATS_DIR = ROOT / 'results' / 'stats'
STATS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load fit ──────────────────────────────────────────────────────────────────
print("Loading fit...")
fitted_models, data = load_fitted_models(str(SAVE_DIR), load_data=True)
print(f"  Models: {list(fitted_models.keys())}")
print(f"  Data: {len(data)} trials, {data['subj'].nunique()} subjects")

fitter = fitted_models['FET_Exp_Bias']

# ── WAIC ──────────────────────────────────────────────────────────────────────
print("\nComputing WAIC...")
waic = compute_waic(fitter, data, n_draws=500)
print(f"  WAIC:   {waic['WAIC']:.2f}  (SE: {waic['se']:.2f})")
print(f"  p_waic: {waic['p_waic']:.2f}")
print(f"  lppd:   {waic['lppd']:.2f}")

# ── PPC predictions ───────────────────────────────────────────────────────────
print("\nGenerating PPC predictions...")
ppc = PosteriorPredictive(fitter)
pred_df = ppc.predict(data, n_draws=500)
metrics = ppc.compute_fit_metrics(data, pred_df)
ppc.print_metrics(metrics, 'FET_Exp_Bias')

# ── Subject-level metrics ─────────────────────────────────────────────────────
print("\nComputing subject metrics...")
subj_metrics = ppc.compute_subject_metrics(data, pred_df)

# ── Population parameters ─────────────────────────────────────────────────────
print("Extracting population parameters...")
pop_params = fitter.get_population_params()
print(pop_params.to_string(index=False))

# ── Subject-level parameters ──────────────────────────────────────────────────
print("\nExtracting subject-level parameters...")
available_params = [k for k in fitter.model.posterior_samples.keys()
                    if k in ['k', 'z', 'beta']]
print(f"  Parameters: {available_params}")

subject_params = {}
for param in available_params:
    subject_params[param] = fitter.get_subject_params(param)

# ── Save results ──────────────────────────────────────────────────────────────
print(f"\nSaving results to {STATS_DIR}...")

# WAIC summary
pd.DataFrame([{
    'Model': 'FET_Exp_Bias',
    'N_subjects': data['subj'].nunique(),
    'N_trials': len(data),
    'WAIC': waic['WAIC'],
    'WAIC_se': waic['se'],
    'p_waic': waic['p_waic'],
    'lppd': waic['lppd'],
    'McFadden_R2': metrics['McFadden_R2'],
    'Accuracy': metrics['Accuracy'],
    'AUC': metrics['AUC'],
    'Brier': metrics['Brier'],
    'ECE': metrics['ECE'],
}]).to_csv(STATS_DIR / 'FET_Exp_Bias_waic.csv', index=False)
print("  Saved: FET_Exp_Bias_waic.csv")

# Subject-level PPC metrics
subj_metrics.to_csv(STATS_DIR / 'FET_Exp_Bias_subject_metrics.csv', index=False)
print("  Saved: FET_Exp_Bias_subject_metrics.csv")

# Predictions
pred_df.to_csv(STATS_DIR / 'FET_Exp_Bias_predictions.csv', index=False)
print("  Saved: FET_Exp_Bias_predictions.csv")

# Population parameters
pop_params.to_csv(STATS_DIR / 'FET_Exp_Bias_population_params.csv', index=False)
print("  Saved: FET_Exp_Bias_population_params.csv")

# Subject-level parameters
for param, df in subject_params.items():
    df.to_csv(STATS_DIR / f'FET_Exp_Bias_{param}_params.csv', index=False)
    print(f"  Saved: FET_Exp_Bias_{param}_params.csv")

print(f"\nDone.")
