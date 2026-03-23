"""
NB03-psych equivalent: Affect × Survival Analysis
Tests whether model-derived survival predicts anxiety and confidence ratings.
Uses L3_add model: S = (1-T) + T/(1+λD), λ≈2.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path('/workspace')
DATA     = BASE / 'data/exploratory_350/processed/stage5_filtered_data_20260320_191950'
STATS    = BASE / 'results/stats'
STATS.mkdir(parents=True, exist_ok=True)

# ── Parameters ─────────────────────────────────────────────────────────────────
LAMBDA   = 2.0   # population-level hyperbolic kernel scale (L3_add)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
feel = pd.read_csv(DATA / 'feelings.csv')

# Rename columns for clarity
feel = feel.rename(columns={
    'attackingProb': 'p_threat',
    'questionLabel': 'affect_type',
    'response':      'rating',
    'trialNumber':   'trial_raw',
    'distanceFromSafety': 'dist_safety',
    'distance':      'dist_level',   # 0/1/2 → D = dist_level + 1
})

# D: maps to model's distance parameter (1/2/3)
feel['D'] = feel['dist_level'] + 1

print(f"Feelings: {len(feel):,} rows, {feel['subj'].nunique()} subjects")
print(f"Affect types: {feel['affect_type'].value_counts().to_dict()}")
print(f"Rating range: {feel['rating'].min()}–{feel['rating'].max()}")
print(f"p_threat levels: {sorted(feel['p_threat'].unique())}")
print(f"D levels: {sorted(feel['D'].unique())}")

# ── 2. Load choice params ──────────────────────────────────────────────────────
params = pd.read_csv(STATS / 'unified_3param_clean.csv')
# unified_3param_clean has: subj, k, beta, alpha
# 'alpha' here is from the SVI fit (not vigor HBM alpha)
print(f"\nParams: {len(params)} subjects, cols: {params.columns.tolist()}")
print(params.describe().round(3))

# Merge params into feelings
feel = feel.merge(params[['subj', 'k', 'beta']], on='subj', how='inner')
print(f"After merge: {len(feel):,} rows, {feel['subj'].nunique()} subjects")

# ── 3. Compute trial-level survival proxy (L3_add) ────────────────────────────
# S = (1-T) + T / (1 + λ·D)
feel['S_probe'] = (1 - feel['p_threat']) + feel['p_threat'] / (1 + LAMBDA * feel['D'])

print(f"\nS_probe range: {feel['S_probe'].min():.3f}–{feel['S_probe'].max():.3f}")
print("S_probe by threat × D:")
print(feel.groupby(['p_threat', 'D'])['S_probe'].mean().round(3).to_string())

# Z-score predictors for LMM
for col in ['p_threat', 'dist_safety', 'S_probe', 'k', 'beta']:
    feel[f'{col}_z'] = (feel[col] - feel[col].mean()) / feel[col].std()

# ── 4. Separate anxiety and confidence ────────────────────────────────────────
anxiety    = feel[feel['affect_type'] == 'anxiety'].copy()
confidence = feel[feel['affect_type'] == 'confidence'].copy()

print(f"\nAnxiety rows:    {len(anxiety):,}")
print(f"Confidence rows: {len(confidence):,}")

# ── 5. LMMs: affect ~ S_probe + (1|subj) ─────────────────────────────────────
print("\n" + "="*60)
print("LMM: affect ~ S_probe_z + (1|subj)")
print("="*60)

results = []

for label, df in [('anxiety', anxiety), ('confidence', confidence)]:
    # Core: rating ~ S_probe_z + (1|subj)
    md = smf.mixedlm("rating ~ S_probe_z", df, groups=df['subj'])
    mdf = md.fit(reml=False, method='lbfgs')

    coef   = mdf.params['S_probe_z']
    se     = mdf.bse['S_probe_z']
    tval   = mdf.tvalues['S_probe_z']
    pval   = mdf.pvalues['S_probe_z']

    n_subj = df['subj'].nunique()
    n_obs  = len(df)

    print(f"\n{label.capitalize()}:")
    print(f"  S_probe_z β={coef:.4f}, SE={se:.4f}, t={tval:.3f}, p={pval:.4f}")
    print(f"  N_subj={n_subj}, N_obs={n_obs}")

    results.append({
        'outcome':    label,
        'predictor':  'S_probe_z',
        'model':      'L3_add (lambda=2.0)',
        'beta':       coef,
        'se':         se,
        't':          tval,
        'p':          pval,
        'n_subj':     n_subj,
        'n_obs':      n_obs,
        'llf':        mdf.llf,
        'aic':        mdf.aic,
        'bic':        mdf.bic,
    })

    # Also run threat model for reference
    md2 = smf.mixedlm("rating ~ p_threat_z + dist_safety_z", df, groups=df['subj'])
    mdf2 = md2.fit(reml=False, method='lbfgs')

    for pred in ['p_threat_z', 'dist_safety_z']:
        results.append({
            'outcome':    label,
            'predictor':  pred,
            'model':      'threat+dist',
            'beta':       mdf2.params[pred],
            'se':         mdf2.bse[pred],
            't':          mdf2.tvalues[pred],
            'p':          mdf2.pvalues[pred],
            'n_subj':     n_subj,
            'n_obs':      n_obs,
            'llf':        mdf2.llf,
            'aic':        mdf2.aic,
            'bic':        mdf2.bic,
        })
    print(f"  p_threat_z β={mdf2.params['p_threat_z']:.4f}, p={mdf2.pvalues['p_threat_z']:.4f}")
    print(f"  dist_safety_z β={mdf2.params['dist_safety_z']:.4f}, p={mdf2.pvalues['dist_safety_z']:.4f}")

# FDR correction across p-values
results_df = pd.DataFrame(results)
reject, pvals_adj, _, _ = multipletests(results_df['p'].values, method='fdr_bh')
results_df['p_fdr'] = pvals_adj
results_df['sig_fdr'] = reject

print("\n" + "="*60)
print("All LMM results with FDR correction:")
print(results_df[['outcome','predictor','beta','se','t','p','p_fdr','sig_fdr']].to_string(index=False))

# ── 6. State-trait decomposition ──────────────────────────────────────────────
print("\n" + "="*60)
print("State-trait decomposition: mean affect ~ k + β (between-subjects OLS)")
print("="*60)

# Compute subject means
trait_df = feel.groupby('subj').agg(
    mean_anxiety    = ('rating', lambda x: x[feel.loc[x.index, 'affect_type'] == 'anxiety'].mean() if (feel.loc[x.index, 'affect_type'] == 'anxiety').any() else np.nan),
    mean_confidence = ('rating', lambda x: x[feel.loc[x.index, 'affect_type'] == 'confidence'].mean() if (feel.loc[x.index, 'affect_type'] == 'confidence').any() else np.nan),
).reset_index()

# Better approach: compute separately
anx_mean  = anxiety.groupby('subj')['rating'].mean().reset_index().rename(columns={'rating': 'mean_anxiety'})
conf_mean = confidence.groupby('subj')['rating'].mean().reset_index().rename(columns={'rating': 'mean_confidence'})

trait_df = params[['subj', 'k', 'beta']].merge(anx_mean, on='subj').merge(conf_mean, on='subj')

# Z-score for OLS
for col in ['k', 'beta', 'mean_anxiety', 'mean_confidence']:
    trait_df[f'{col}_z'] = (trait_df[col] - trait_df[col].mean()) / trait_df[col].std()

trait_results = []

for outcome in ['mean_anxiety', 'mean_confidence']:
    formula = f"{outcome}_z ~ k_z + beta_z"
    ols = smf.ols(formula, data=trait_df).fit()

    r2    = ols.rsquared
    r2adj = ols.rsquared_adj
    fstat = ols.fvalue
    fpval = ols.f_pvalue

    print(f"\n{outcome} ~ k + β  (N={len(trait_df)}):")
    print(f"  R²={r2:.4f}, R²adj={r2adj:.4f}, F={fstat:.3f}, p={fpval:.4f}")

    for pred in ['k_z', 'beta_z']:
        coef  = ols.params[pred]
        se    = ols.bse[pred]
        tval  = ols.tvalues[pred]
        pval  = ols.pvalues[pred]
        print(f"  {pred}: β={coef:.4f}, SE={se:.4f}, t={tval:.3f}, p={pval:.4f}")
        trait_results.append({
            'outcome':  outcome,
            'predictor': pred,
            'beta':     coef,
            'se':       se,
            't':        tval,
            'p':        pval,
            'r2_model': r2,
            'r2adj':    r2adj,
            'f_model':  fstat,
            'f_p':      fpval,
            'n':        len(trait_df),
        })

trait_results_df = pd.DataFrame(trait_results)

# ── 7. Cross-domain summary: S_probe × affect ─────────────────────────────────
print("\n" + "="*60)
print("Cross-domain correlations: mean S_probe ~ mean affect")
print("="*60)

# Compute per-subject mean S_probe
s_mean = feel.groupby('subj')['S_probe'].mean().reset_index().rename(columns={'S_probe': 'mean_S_probe'})
trait_df = trait_df.merge(s_mean, on='subj')

for affect in ['mean_anxiety', 'mean_confidence']:
    r, p = stats.pearsonr(trait_df['mean_S_probe'], trait_df[affect])
    print(f"  mean_S_probe ~ {affect}: r={r:.4f}, p={p:.4f}")

# ── 8. Save results ───────────────────────────────────────────────────────────
out_lmm = STATS / 'affect_lmm_results.csv'
results_df.to_csv(out_lmm, index=False)
print(f"\nSaved LMM results → {out_lmm}")

out_trait = STATS / 'affect_trait_scores.csv'
trait_df.to_csv(out_trait, index=False)
print(f"Saved trait scores → {out_trait}")

print("\nDone.")
