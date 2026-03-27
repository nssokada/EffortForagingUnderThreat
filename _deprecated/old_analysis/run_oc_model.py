"""
Run the Optimal Control choice model via SVI.

Usage:
    python scripts/run_oc_model.py

Fits c_effort and c_death per subject using the OC framework,
then evaluates choice accuracy and predicted vigor (u*).
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from scripts.modeling.optimal_control import prepare_trial_data
from scripts.modeling.oc_model import fit_oc_svi, extract_params

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = ROOT / 'data' / 'exploratory_350' / 'processed'
STAGE5 = sorted(DATA_DIR.glob('stage5_filtered_data_*'))[-1]
OUT_DIR = ROOT / 'results' / 'stats'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
beh = pd.read_csv(STAGE5 / 'behavior_rich.csv')
print(f"  {beh['subj'].nunique()} subjects, {len(beh)} trials")

# ── Prepare for model ─────────────────────────────────────────────────────────
print("\nPreparing trial data...")
data = prepare_trial_data(beh)
print(f"  {data['n_subjects']} subjects, {len(data['T'])} trials")

# ── Fit SVI ───────────────────────────────────────────────────────────────────
result = fit_oc_svi(
    data,
    n_steps=30000,
    lr=0.002,
    guide_type='autonormal',
    print_every=5000,
)

# ── Extract parameters ────────────────────────────────────────────────────────
print("\nExtracting parameters...")
params = extract_params(result, n_samples=2000)

print("\nPopulation parameters:")
for name, vals in params['population'].items():
    print(f"  {name:15s}: {vals['mean']:.4f} ± {vals['std']:.4f}")

subj_df = params['subject']
print(f"\nSubject parameters (n={len(subj_df)}):")
print(f"  c_effort: mean={subj_df['c_effort_mean'].mean():.3f}, "
      f"range=[{subj_df['c_effort_mean'].min():.3f}, {subj_df['c_effort_mean'].max():.3f}]")
print(f"  c_death:  mean={subj_df['c_death_mean'].mean():.3f}, "
      f"range=[{subj_df['c_death_mean'].min():.3f}, {subj_df['c_death_mean'].max():.3f}]")

# ── Save ──────────────────────────────────────────────────────────────────────
subj_df.to_csv(OUT_DIR / 'oc_subject_params.csv', index=False)
print(f"\nSaved: {OUT_DIR / 'oc_subject_params.csv'}")

# ── Choice accuracy ───────────────────────────────────────────────────────────
print("\n── Choice Accuracy ──")
# Get predicted choice probabilities
from numpyro.infer import Predictive
from jax import random

predictive = Predictive(result['guide'], params=result['params'], num_samples=500)
samples = predictive(random.PRNGKey(44), **result['model_args'])
p_H_samples = np.array(samples['p_H'])  # (500, n_trials)
p_H_mean = p_H_samples.mean(axis=0)

choice = np.array(data['choice'])
pred_choice = (p_H_mean > 0.5).astype(int)
accuracy = (pred_choice == choice).mean()
print(f"  Overall accuracy: {accuracy:.3f}")

# By trial type
threat = np.array(data['T'])
for t in [0.1, 0.5, 0.9]:
    mask = np.isclose(threat, t)
    acc = (pred_choice[mask] == choice[mask]).mean()
    print(f"  T={t}: accuracy={acc:.3f} (n={mask.sum()})")

# ── Predicted vigor (u*) ─────────────────────────────────────────────────────
print("\n── Predicted Vigor ──")
u_H_samples = np.array(samples['u_star_H'])
u_L_samples = np.array(samples['u_star_L'])
u_H_mean = u_H_samples.mean(axis=0)
u_L_mean = u_L_samples.mean(axis=0)

# Predicted vigor for chosen option
u_chosen = np.where(choice == 1, u_H_mean, u_L_mean)
print(f"  Mean predicted u* (chosen): {u_chosen.mean():.3f}")
print(f"  SD predicted u*: {u_chosen.std():.3f}")
print(f"  Range: [{u_chosen.min():.3f}, {u_chosen.max():.3f}]")

# Compare to observed vigor if available
if 'mean_trial_effort' in beh.columns:
    observed_vigor = beh['mean_trial_effort'].values
    from scipy.stats import pearsonr
    mask = ~np.isnan(observed_vigor)
    r, p = pearsonr(u_chosen[mask], observed_vigor[mask])
    print(f"\n  Correlation u* vs observed vigor: r={r:.3f}, p={p:.2e}")

# ── Correlation between c_effort and c_death ──────────────────────────────────
from scipy.stats import pearsonr
r, p = pearsonr(subj_df['c_effort_mean'], subj_df['c_death_mean'])
print(f"\n  c_effort × c_death: r={r:.3f}, p={p:.2e}")

# ── Save loss curve ───────────────────────────────────────────────────────────
losses = result['losses']
loss_df = pd.DataFrame({'step': range(len(losses)), 'loss': losses})
loss_df.to_csv(OUT_DIR / 'oc_svi_losses.csv', index=False)

print(f"\n{'='*60}")
print("OC model fitting complete.")
print(f"{'='*60}")
